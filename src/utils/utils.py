import os
from typing import Tuple, Union

import torch
import yaml
from torch_geometric.data import Data, HeteroData


def choice(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return t.select(dim, int(torch.randint(0, t.size(0), (1,))))


def falling_factorial(x: int, N: int) -> int:
    assert x >= N, "x is smaller than N"
    result = 1
    for n in range(N):
        result *= x - n
    return result


def load_ds_meta(data_path: str) -> Tuple[int, int, int]:
    with open(os.path.join(data_path, "meta.yml")) as f:
        D = yaml.safe_load(f)

    node_features = D["node_features"]
    edge_features = D["edge_features"]
    return node_features, edge_features


def encode_type_as_feature(graph: Union[HeteroData, Data]) -> Data:
    """Encodes pytorch geometric data objects to use node / edge type as binary feature vector

    Args:
        graph (Union[HeteroData, Data]): graph

    Returns:
        Data: graph with re-encoded type feature vectors
    """
    if isinstance(graph, HeteroData):
        node_feature_length = len(graph.node_types)
        edge_feature_length = len(graph.edge_types)

        for i, _ in enumerate(graph.node_types):
            nodes_of_type = graph.node_stores[i].num_nodes
            graph.node_stores[i].x = torch.zeros((nodes_of_type, node_feature_length))
            graph.node_stores[i].x[:, i] = 1

        for i, _ in enumerate(graph.edge_types):
            edges_of_type = len(graph.edge_stores[i].edge_index[0])
            graph.edge_stores[i].edge_attr = torch.zeros(
                (edges_of_type, edge_feature_length)
            )
            graph.edge_stores[i].edge_attr[:, i] = 1

        graph = graph.to_homogeneous(dummy_values=False)

    elif isinstance(graph, Data):
        graph.x = torch.ones((graph.num_nodes, 1))
        graph.edge_attr = torch.ones((graph.num_edges, 1))

    return graph


def min_max_normalize(input: torch.Tensor) -> torch.Tensor:
    m = input.min()
    M = input.max()
    return (input - m) / (M - m)
