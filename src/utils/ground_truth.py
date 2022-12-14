import multiprocessing.dummy
import os
from multiprocessing.pool import ThreadPool
from typing import List, Optional, Tuple

import torch
from networkx.algorithms import isomorphism as iso
from rich.progress import Progress, TextColumn, track
from torch_geometric import utils
from torch_geometric.data.data import Data

from utils.rdf import RDFRW, SparqlQueryEngine
from utils.samplers import IndexedSubgraph, NoNeighborsException, Sampler

PARALLEL_SPARQL_QUERIES = os.cpu_count()


def data_to_nx(data: Data):
    return utils.to_networkx(data, ["x"], ["edge_attr"])


def node_equal(node1, node2):
    return node1["x"] == node2["x"]


def edge_equal(edge1, edge2):
    return edge1["edge_attr"] == edge2["edge_attr"]


def iso_match(g, K_ij):
    matcher = iso.DiGraphMatcher(g, K_ij, node_match=node_equal, edge_match=edge_equal)
    matches = matcher.subgraph_isomorphisms_iter()
    return len(list(matches))


class GroundTruth:
    def __init__(self, endpoint: str) -> None:
        self.query_engine = SparqlQueryEngine(endpoint)

    def match_test(self, graph_data: Data, K_i: List[IndexedSubgraph]) -> torch.Tensor:
        """Used to validate SPARQL queries against networkx subgraph isomorphism
        counts.

        Args:
            graph_data (Data): graph
            K_i (List[IndexedSubgraph]): list of sampled patterns

        Returns:
            torch.Tensor: support values counted using networkx
        """
        results = []
        queries = []

        K_i = [graph_data.subgraph(e.node_indices) for e in K_i]
        graph_data = data_to_nx(graph_data)
        K_i = [data_to_nx(e) for e in K_i]

        pool: ThreadPool = multiprocessing.dummy.Pool(PARALLEL_SPARQL_QUERIES)
        for K_ij in K_i:
            queries.append(pool.apply_async(iso_match, [graph_data, K_ij]))

        for query in track(queries, "Running queries"):
            value = query.get()
            results.append(value)

        return torch.tensor(results).unsqueeze(1)

    def __call__(self, graph_data: Data, K_i: List[IndexedSubgraph]) -> torch.Tensor:
        """Counts pattern support in a graph using SPARQL queries

        Args:
            graph_data (Data): a graph
            K_i (List[IndexedSubgraph]): list of sampled patterns

        Returns:
            torch.Tensor: support values for each pattern
        """
        rdf_rw = RDFRW()
        rdf_g = rdf_rw.torch_data_to_rdflib(graph_data)

        self.query_engine.insert(rdf_g)

        results = []
        queries = []

        # Pool of async threads, one for each K_ij, query the server.
        pool: ThreadPool = multiprocessing.dummy.Pool(PARALLEL_SPARQL_QUERIES)
        for K_ij in K_i:
            queries.append(pool.apply_async(self._support, [graph_data, K_ij]))

        timed_out = 0
        with Progress(
            *Progress.get_default_columns(),
            TextColumn("Timed out: {task.fields[timed_out]}"),
        ) as bar:
            query_task = bar.add_task(
                "Running queries", total=len(queries), timed_out=0
            )
            for n, query in enumerate(queries):
                value = query.get()
                if value == -1:
                    timed_out += 1
                results.append(value)
                bar.update(query_task, completed=n + 1, timed_out=timed_out)

        return torch.tensor(results).unsqueeze(1)

    def _support(self, graph_data: Data, K_ij: IndexedSubgraph) -> float:
        """Returns the support of a pattern in a graph using a sparql query

        Args:
            graph_data (Data): graph
            K_ij (IndexedSubgraph): pattern

        Returns:
            float: the support value
        """
        sparql_query = self.query_engine.generate_query(graph_data, K_ij)
        supp = self.query_engine.run_query(sparql_query)
        assert supp != 0, "Malformed query, support 0"

        return supp


def generate_training_samples(
    g_i: Data,
    ground_truth: GroundTruth,
    k: int,
    k_sampler: Sampler,
    max_nr_of_samples: int,
) -> Tuple:
    """Generates training samples by sampling patterns, creating sparql queries for the patterns,
    executing them in graph g_i, and returning the resulting 'subgraph, pattern, support' tuples.

    Args:
        g_i (Data): subgraph input
        ground_truth (GroundTruth): class or function that evaluates ground truth
        k (int): allowed size of patterns
        k_sampler (Sampler): class or function for sampling patterns
        max_nr_of_samples (int): nr of patterns to sample and evaluate

    Returns:
        Tuple: 'subgraph, patterns, support values' tuple
    """
    # Will sample up max_nr_of_samples samples and discard invalid ones (those
    # that are too small or timed out).

    K_i = sample_k(g_i, k, k_sampler, max_nr_of_samples)
    supp = ground_truth(g_i, K_i)

    # Discard samples that timed out.
    mask: torch.BoolTensor = (supp != -1).flatten()

    K_i_indices = torch.stack([e.node_indices for e in K_i])
    K_i_indices = K_i_indices[mask]
    supp = supp[mask]

    return g_i, K_i_indices, supp


def sample_k(
    g_i: Data, k: int, k_sampler: Sampler, samples_in_batch: int
) -> List[IndexedSubgraph]:
    K_i = []
    K_i = k_sampler(g_i, k, torch.randint(g_i.num_nodes, (samples_in_batch,)))

    # Remove samples that are too small.
    indices_to_keep = []
    for j, K_ij in enumerate(K_i):
        if K_ij.node_indices.size(0) == k:
            indices_to_keep.append(j)

    K_i = [K_i[i] for i in indices_to_keep]

    if len(K_i) == 0:
        raise NoNeighborsException(f"No subgraphs contain {k} neighbors")

    return K_i
