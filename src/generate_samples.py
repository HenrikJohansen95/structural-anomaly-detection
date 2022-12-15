import typer


class NonExistantDataset(Exception):
    pass


def generate_samples(
    dataset: str = typer.Argument(..., help="name of dataset"),
    save_path: str = typer.Argument(..., help="where to save generated samples"),
    k: int = typer.Argument(..., help="size of patterns"),
    graphs_per_size: int = typer.Argument(
        ..., help="how many subgraphs of each size to sample"
    ),
    K_per_graph: int = typer.Argument(
        ..., help="how many patterns to sample from each subgraph"
    ),
) -> None:
    """Generates training samples from PyTorch Geometric datasets

    Args:
        dataset (str): name of dataset
        save_path (str): where to save generated samples
        k (int): size of patterns
        graphs_per_size (int): how many subgraphs of each size to sample
        K_per_graph (int): how many patterns to sample from each subgraph

    Raises:
        NonExistantDataset: raised if the dataset name is invalid
    """
    # Lazy imports
    from pathlib import Path

    import torch
    import yaml
    from torch_geometric.data import Data
    from torch_geometric.datasets import IMDB, LastFM, Planetoid

    import config
    from utils.ground_truth import GroundTruth, generate_training_samples
    from utils.samplers import (
        KNeighborsSampler,
        NoNeighborsException,
        SampleRandomConnectedSubgraph,
    )
    from utils.utils import encode_type_as_feature

    # Load dataset
    if dataset == "cora":
        data = encode_type_as_feature(Planetoid("rawdata/cora", "cora", "full").data)
        splits = {"train": [512, 1024, 2048], "validate": [4096], "test": [4096]}
    elif dataset == "lastfm":
        data = encode_type_as_feature(LastFM("rawdata/lastfm").data)
        splits = {"train": [256, 512, 768], "validate": [1024], "test": [1024]}
    elif dataset == "imdb":
        data = encode_type_as_feature(IMDB("rawdata/imdb").data)
        splits = {"train": [512, 1024, 2048], "validate": [4096], "test": [4096]}
    else:
        raise NonExistantDataset

    for split in splits:
        sizes = splits[split]
        nr = 0
        tuples = []
        for i, size in enumerate(sizes):
            indexed_subgraphs = KNeighborsSampler()(
                data,
                size,
                torch.randint(0, data.num_nodes, (graphs_per_size,)),
                relabel_nodes=True,
            )
            # Remove the subgraphs that are too small
            for i, g_i in reversed(list(enumerate(indexed_subgraphs))):
                if g_i.node_indices.size(-1) < size:
                    indexed_subgraphs.pop(i)

            # Go to next size if there are no subgraphs left
            if len(indexed_subgraphs) < 1:
                print(f"No subgraphs of size {size}")
                continue
            for g_i in indexed_subgraphs:
                g_i = Data(data.x[g_i.node_indices], g_i.edge_index, g_i.edge_attr)

                print("split", split)
                print("nr", nr)
                print("nodes", g_i.num_nodes)
                print("edges", g_i.num_edges)

                try:
                    _, K_i, supp_i = generate_training_samples(
                        g_i,
                        GroundTruth(config.SPARQL_ENDPOINT),
                        k,
                        SampleRandomConnectedSubgraph(),
                        K_per_graph,
                    )
                except NoNeighborsException as e:
                    print(e)
                    print("Moving on to next subgraph")

                tuples.append((g_i, K_i, supp_i))
                nr += 1
                print()

        path = f"{save_path}/k{k}/{dataset}"
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(tuples, f"{path}/{split}.pt")

    with open(f"{save_path}/k{k}/{dataset}/meta.yml", "w") as f:
        yaml.safe_dump(
            {
                "node_features": g_i.num_node_features,
                "edge_features": g_i.num_edge_features,
                "k": k,
                "splits": splits,
                "graphs_per_size": graphs_per_size,
                "K_per_graph": K_per_graph,
            },
            f,
        )


if __name__ == "__main__":
    typer.run(generate_samples)
