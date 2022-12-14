from __future__ import annotations

from typing import List

import torch
from torch_geometric.data import Data


class NoNeighborsException(Exception):
    pass


class IndexedSubgraph:
    """Stores node indices along with their original edge connections and edge
    attributes."""

    def __init__(
        self,
        node_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> None:
        self.node_indices = node_indices
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __repr__(self) -> str:
        return f"IndexedSubgraph(node_indices={self.node_indices}, edge_index={self.edge_index}, edge_attr={self.edge_attr})"


class Sampler:
    def __call__(
        self, data: Data, k: int, start_indices: torch.LongTensor, relabel_nodes=False
    ) -> List[IndexedSubgraph]:
        raise NotImplementedError


class KNeighborsSampler(Sampler):
    def __call__(
        self, data: Data, k: int, start_indices: torch.LongTensor, relabel_nodes=False
    ) -> List[IndexedSubgraph]:
        """Basically a rewrite of torch_geometric.utils.k_hop_subgraph that
        samples k nodes instead of all nodes in k hops.

        Args:
            data (Data): graph
            k (int): size of subgraphs to be sampled
            start_indices (torch.LongTensor): indices to start sampling from (i.e. the first node in the neighborhood)
            relabel_nodes (bool, optional): whether to change the indices to start from 0 again, or keep the indices from the original graph. Defaults to False.

        Returns:
            List[IndexedSubgraph]: list of sampled subgraphs / patterns
        """

        samples: List[IndexedSubgraph] = []

        from_nodes: torch.Tensor
        to_nodes: torch.Tensor
        from_nodes, to_nodes = data.edge_index

        for start_index in start_indices.unsqueeze(-1):
            node_mask: torch.Tensor = from_nodes.new_empty(
                data.num_nodes, dtype=torch.bool
            )
            edge_mask: torch.Tensor = from_nodes.new_empty(
                from_nodes.size(0), dtype=torch.bool
            )

            node_indices_list: List[torch.Tensor] = [start_index]
            concatenated_indices = start_index
            node_count = 1
            prev_node_count = 0

            while node_count < k and prev_node_count != node_count:
                node_mask.fill_(False)
                node_mask[node_indices_list[-1]] = True
                torch.index_select(node_mask, 0, from_nodes, out=edge_mask)
                node_indices_list.append(to_nodes[edge_mask])

                concatenated_indices = torch.cat(
                    (concatenated_indices, node_indices_list[-1])
                )

                prev_node_count = node_count
                node_count = concatenated_indices.unique().size(0)

            node_indices: torch.Tensor
            overflow = node_count - k

            if overflow > 0:
                last_iteration: torch.Tensor = node_indices_list[-1].unique()
                node_indices = torch.cat(node_indices_list[:-1]).unique()

                mask = last_iteration.new_empty(last_iteration.size())
                mask.fill_(False)

                for i in node_indices:
                    mask |= last_iteration == i
                mask = mask == False

                last_iteration = last_iteration[mask][:-overflow]
                node_indices = torch.cat((node_indices, last_iteration))
            else:
                node_indices = torch.cat(node_indices_list).unique()

            node_mask.fill_(False)
            node_mask[node_indices] = True
            edge_mask = node_mask[from_nodes] & node_mask[to_nodes]

            edge_index: torch.Tensor = data.edge_index[:, edge_mask]
            edge_attr: torch.Tensor = data.edge_attr[edge_mask]

            if relabel_nodes:
                node_idx = from_nodes.new_full((data.num_nodes,), -1)
                node_idx[node_indices] = torch.arange(
                    node_indices.size(0), device=from_nodes.device
                )
                edge_index = node_idx[edge_index]

            samples.append(IndexedSubgraph(node_indices, edge_index, edge_attr))

        return samples


class SampleRandomConnectedSubgraph(Sampler):
    def __call__(
        self, data: Data, k: int, start_indices: torch.LongTensor, relabel_nodes=False
    ) -> List[IndexedSubgraph]:
        """Samples subgraphs that are connected, but each new node is sampled randomly instead of as a uniformely growing neighborhood.

        Args:
            data (Data): graph
            k (int): size of subgraphs to be sampled
            start_indices (torch.LongTensor): indices to start sampling from (i.e. the first node in the neighborhood)
            relabel_nodes (bool, optional): whether to change the indices to start from 0 again, or keep the indices from the original graph. Defaults to False.

        Returns:
            List[IndexedSubgraph]: list of sampled subgraphs / patterns
        """
        samples: List[IndexedSubgraph] = []
        from_nodes: torch.Tensor
        to_nodes: torch.Tensor
        from_nodes, to_nodes = data.edge_index

        for start_index in start_indices.unsqueeze(-1):
            # Nodes that have been added
            node_indices_list = [int(start_index)]

            node_mask: torch.Tensor = from_nodes.new_empty(
                data.num_nodes, dtype=torch.bool
            )
            edge_mask: torch.Tensor = from_nodes.new_empty(
                from_nodes.size(0), dtype=torch.bool
            )

            while len(node_indices_list) < k:
                node_mask.fill_(False)
                node_mask[node_indices_list] = True

                torch.index_select(node_mask, 0, from_nodes, out=edge_mask)

                neighbors = to_nodes[edge_mask].tolist()
                new_neighbors = list(set(neighbors) - set(node_indices_list))

                if len(new_neighbors) == 0:
                    break
                else:
                    random_new_neighbor = new_neighbors[
                        torch.randint(len(new_neighbors), (1,))
                    ]
                    node_indices_list.append(random_new_neighbor)

            node_indices = torch.tensor(node_indices_list)
            node_mask.fill_(False)
            node_mask[node_indices] = True
            edge_mask = node_mask[from_nodes] & node_mask[to_nodes]

            edge_index: torch.Tensor = data.edge_index[:, edge_mask]
            edge_attr: torch.Tensor = data.edge_attr[edge_mask]

            if relabel_nodes:
                node_idx = from_nodes.new_full((data.num_nodes,), -1)
                node_idx[node_indices] = torch.arange(
                    node_indices.size(0), device=from_nodes.device
                )
                edge_index = node_idx[edge_index]

            samples.append(IndexedSubgraph(node_indices, edge_index, edge_attr))

        return samples
