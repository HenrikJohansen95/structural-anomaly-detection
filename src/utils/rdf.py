import requests
import torch
from rdflib import OWL, RDF, RDFS, Graph, Literal, Namespace, URIRef
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

from utils.samplers import IndexedSubgraph


class SparqlQueryEngine:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def _get_node_types(self, node_features, node_indices):
        # for n in K_j get classes
        node_types = []
        negated_types = []
        for node_idx in node_indices:
            features = node_features[node_idx]
            # type_string = "".join(str(e) for e in features.long().tolist())
            for type_idx, type_feature in enumerate(features):
                if type_feature.item() == 1:
                    node_types.append(f"?{node_idx} a ex:node_type_{type_idx} .")
                elif type_feature.item() == 0:
                    negated_types.append(
                        f"MINUS {{ ?{node_idx} a ex:node_type_{type_idx} . }}"
                    )
        return node_types, negated_types

    def _get_node_inequalities(self, node_features, node_indices):
        # for n0, n1 in K_J add n0 =! n1
        inequalities = []
        for i in range(node_indices.size(0)):
            i_index = node_indices[i]
            node_0 = node_features[i_index]
            for j in range(i + 1, node_indices.size(0)):
                j_index = node_indices[j]
                node_1 = node_features[j_index]
                if node_0.equal(node_1):
                    inequalities.append(f"!sameTerm(?{i_index}, ?{j_index})")

        return inequalities

    def _get_edge_types(self, edge_index, edge_features, node_indices):
        # for P(n0, n1) in edge_index add P(n0, n1)
        edges = []
        negated_edges = []

        for i, edge in enumerate(edge_index.t()):
            features = edge_features[i]
            # type_string = "".join(str(e) for e in features.long().tolist())
            for edge_idx, type_feature in enumerate(features):
                if type_feature.item() == 1:
                    edges.append(
                        f"?{edge[0].item()} ex:edge_type_{edge_idx} ?{edge[1].item()} ."
                    )

        for node_idx_0 in node_indices:
            for node_idx_1 in node_indices:
                for edge_idx in range(edge_features.size(1)):
                    statement = f"?{node_idx_0} ex:edge_type_{edge_idx} ?{node_idx_1} ."
                    if statement not in edges:
                        negated_edges.append(f"MINUS {{ {statement} }}")

        return edges, negated_edges

    def _format_query_strings(
        self, node_types, predicates, inequalities, negated_node_types, negated_edges
    ):
        node_types = "\n\t" + "\n\t".join(node_types)
        predicates = "\n\t" + "\n\t".join(predicates) if len(predicates) > 0 else ""

        if len(inequalities) > 0:
            tmp = "\n\tFILTER (\n\t\t"
            tmp += " &&\n\t\t".join(inequalities)
            tmp += "\n\t)"
            inequalities = tmp
        else:
            inequalities = ""

        negations = "\n\t"
        if len(negated_node_types) > 0:
            negations += "\n\t".join(negated_node_types)
            negations += "\n\t"
        if len(negated_edges) > 0:
            negations += "\n\t".join(negated_edges)

        return node_types, predicates, inequalities, negations

    def generate_query(self, graph_data: Data, K_ij: IndexedSubgraph) -> str:
        """Generates a SPARQL query for a pattern in a graph.

        Args:
            graph_data (Data): graph
            K_ij (IndexedSubgraph): pattern


        Returns:
            str: SPARQL query string
        """
        node_types, negated_node_types = self._get_node_types(
            graph_data.x, K_ij.node_indices
        )
        inequalities = self._get_node_inequalities(graph_data.x, K_ij.node_indices)
        edges, negated_edges = self._get_edge_types(
            K_ij.edge_index, K_ij.edge_attr, K_ij.node_indices
        )

        node_types, edges, inequalities, negations = self._format_query_strings(
            node_types, edges, inequalities, negated_node_types, negated_edges
        )

        query_string = f"""\
PREFIX ex: <http://example.org/>

SELECT (count(*) as ?count) WHERE {{ {node_types} {edges} {inequalities} {negations}
}}"""
        return query_string

    def insert(self, rdf_g: Graph):
        """Uploads a rdflib.Graph to a SPARQL server

        Args:
            rdf_g (Graph): rdflib.Graph
        """
        ttl_string = rdf_g.serialize(None, "ttl")

        # Convert from .ttl to sparql format.
        prefix, ttl_string = ttl_string.split("\n", 1)
        prefix = prefix[:-1][1:] + "\n"
        graph_insert_string = f"{prefix}INSERT DATA {{{ttl_string}}}"

        try:
            requests.post(self.endpoint, {"update": "CLEAR ALL;" + graph_insert_string})
        except requests.exceptions.ConnectionError:
            print(f"Could not connect to SPAQRL endpoint: {self.endpoint}")
            quit()

    def run_query(self, query: str) -> int:
        """Runs a SPARQL query against the SPARQL server (Jena).

        Args:
            query (str): query string

        Raises:
            Exception: exception raised if SPARQL endpoint is not reachable

        Returns:
            int: support for the query
        """
        r = requests.get(self.endpoint, {"query": query})
        if r.status_code == 503 or r.status_code == 500:
            return -1
        elif r.status_code == 404:
            raise Exception(f"SPARQL ENDPOINT: Error {r.status_code}")
        elif r.status_code == 200:
            bindings = r.json()["results"]["bindings"]
            return int(bindings[0]["count"]["value"])


class RDFRW:
    def torch_data_to_rdflib(self, graph: Data) -> Graph:
        """Converts pytorch geometric Data objects to rdflib.Graphs

        Args:
            graph (Data): pytorch geometric Data object

        Returns:
            Graph: rdflib.Graph
        """
        rdf_g = Graph()
        ex = Namespace("http://example.org/")
        rdf_g.namespace_manager.bind("ex", ex)

        # Get all node indices for node_type
        features: torch.Tensor
        for node_idx, features in enumerate(graph.x):
            # type_string = "".join(str(e) for e in features.long().tolist())
            node_i = URIRef(f"{ex}node_{node_idx}")
            for type_idx, type_feature in enumerate(features):
                if type_feature.item() == 1:
                    node_type = URIRef(f"{ex}node_type_{type_idx}")
                    rdf_g.add((node_i, RDF.type, node_type))

        # Get all edge indices for edge_type
        edge: torch.Tensor
        for edge_idx, edge in enumerate(graph.edge_index.t()):
            features = graph.edge_attr[edge_idx]
            # type_string = "".join(str(e) for e in features.long().tolist())

            from_node = URIRef(f"{ex}node_{edge[0]}")
            to_node = URIRef(f"{ex}node_{edge[1]}")
            for edge_idx, type_feature in enumerate(features):
                if type_feature.item() == 1:
                    edge_type = URIRef(f"{ex}edge_type_{edge_idx}")
                    rdf_g.add((from_node, edge_type, to_node))

        return rdf_g

    def rdf_to_torch_data(self, g: Graph) -> Data:
        """Parses a rdflib.Graph and converts it to a pytorch geometric Data object

        Args:
            g (Graph): the rdflib.Graph

        Returns:
            Data: parsed pytorch geometric Data object with correct edge and node types
        """
        # Remove triples with OWL, RDFS and Literals, leaving just nodes.
        for triple in g.triples((None, None, None)):
            if isinstance(triple[2], Literal):
                g.remove(triple)
            else:
                for e in triple:
                    if e.startswith(RDFS) or e.startswith(OWL):
                        g.remove(triple)
                        break
        node_types = set()
        for s, p, o in g.triples((None, RDF.type, None)):
            node_types.add(o)
        node_type_dict = {}
        num_node_types = len(node_types)

        for i, nt in enumerate(node_types):
            node_type_dict[nt] = i

        edge_types = set()
        num_edges = 0
        for s, p, o in g.triples((None, None, None)):
            # Filter Literal relations
            if isinstance(s, URIRef) and isinstance(o, URIRef) and p != RDF.type:
                edge_types.add(p)
                num_edges += 1
        num_edge_types = len(edge_types)

        edge_type_dict = {}
        for i, et in enumerate(edge_types):
            edge_type_dict[et] = i

        unique_nodes = set()
        for s, p, o in g.triples((None, None, None)):
            if isinstance(s, URIRef):
                unique_nodes.add(s)
            if isinstance(o, URIRef):
                unique_nodes.add(o)

        num_nodes = len(unique_nodes)
        node_dict = {}
        x = torch.zeros((num_nodes, num_node_types))

        for i, node in enumerate(unique_nodes):
            node_dict[node] = i
            for s, p, o in g.triples((node, RDF.type, None)):
                x[i, node_type_dict[o]] = 1

        edge_attr = torch.zeros((num_edges, num_edge_types))
        edge_index = torch.zeros((num_edges, 2), dtype=torch.long)

        i = 0
        for s, p, o in g.triples((None, None, None)):
            if isinstance(s, URIRef) and isinstance(o, URIRef) and p != RDF.type:
                edge_index[i] = torch.tensor((node_dict[s], node_dict[o]))
                edge_attr[i, edge_type_dict[p]] = 1
                i += 1
        edge_index = edge_index.t().contiguous()
        edge_index, edge_attr = coalesce(edge_index, edge_attr, reduce="max")

        return Data(x, edge_index, edge_attr)


def print_data_as_ttl(graph: Data):
    return RDFRW().torch_data_to_rdflib(graph).print()
