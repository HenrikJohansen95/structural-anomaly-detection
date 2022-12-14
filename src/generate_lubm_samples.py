import typer


def generate_lubm_samples(
    base_lubm_path: str = typer.Argument(
        None, help="the parent path of the generated lubm .owl files"
    ),
    save_path: str = typer.Argument(None, help="where to save generated samples"),
    k: int = typer.Argument(None, help="size of patterns"),
    graphs_per_size: int = typer.Argument(
        None, help="how many subgraphs of each size to sample"
    ),
    K_per_graph: int = typer.Argument(
        None, help="how many patterns to sample from each subgraph"
    ),
) -> None:
    """Generates training samples from the LUBM graph.
    With a slight refactor it's probably usable for any .owl graphs,
    but this has not been tested.

    Args:
        base_lubm_path (str): the parent path of the generated lubm .owl files
        save_path (str): where to save generated samples
        k (int): size of patterns
        graphs_per_size (int): how many subgraphs of each size to sample
        K_per_graph (int): how many patterns to sample from each subgraph
    """
    # Lazy imports
    from pathlib import Path

    import torch
    import yaml
    from rdflib import Graph

    import config
    from utils.ground_truth import GroundTruth, generate_training_samples
    from utils.rdf import RDFRW
    from utils.samplers import SampleRandomConnectedSubgraph

    splits = {"train": [1, 2, 4], "validate": [8], "test": [8]}
    owl_paths = list(Path(base_lubm_path).glob("lubm\\University*.owl"))

    assert sum([sum(v) for v in splits.values()]) * graphs_per_size < len(owl_paths)

    offset = 0
    for split in splits:
        tuples = []

        for nd in splits[split]:
            for _ in range(graphs_per_size):
                # Load dataset
                g = Graph()
                for i in range(nd):
                    g.parse(owl_paths[i + offset])
                data = RDFRW().rdf_to_torch_data(g)

                print("split", split)
                print("nodes", data.num_nodes)
                print("edges", data.num_edges)

                _, K, supp = generate_training_samples(
                    data,
                    GroundTruth(config.SPARQL_ENDPOINT),
                    k,
                    SampleRandomConnectedSubgraph(),
                    K_per_graph,
                )

                tuples.append((data, K, supp))
                offset += nd
                print()

        path = f"{save_path}/k{k}/lubm"
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(tuples, f"{path}/{split}.pt")

    with open(f"{save_path}/k{k}/lubm/meta.yml", "w") as f:
        yaml.safe_dump(
            {
                "node_features": data.num_node_features,
                "edge_features": data.num_edge_features,
                "k": k,
                "splits": splits,
                "graphs_per_size": graphs_per_size,
                "K_per_graph": K_per_graph,
            },
            f,
        )


if __name__ == "__main__":
    typer.run(generate_lubm_samples)
