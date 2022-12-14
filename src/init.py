from pathlib import Path

from pl_modules.data_module import GraphDataModule
from pl_modules.system import GraphSystem
from pl_modules.transforms import Standardizer, SupportToAnomScore, ThresholdAtDelta
from utils.utils import load_ds_meta


def init_system(hparams: dict, dir_prefix: str) -> GraphSystem:
    """Initializes our system based on a dictionary of hyperparameters

    Args:
        hparams (dict): dictionary of hyperparameters
        dir_prefix (str): path prefix, used by ray tune mostly

    Returns:
        GraphSystem: a configured system
    """
    k = hparams["k"]
    data_path = Path(dir_prefix, f"data/k{k}/{hparams['datasetname']}")
    node_features, edge_features = load_ds_meta(data_path)

    system = GraphSystem(
        lr=hparams["lr"],
        wd=hparams["wd"],
        pos_sample_weight=hparams["pos_weight"],
        node_features=node_features,
        edge_features=edge_features,
        k=k,
        embed_type=hparams["emb"],
        embed_dim=hparams["emb_dim"],
        embed_n_layers=hparams["emb_depth"],  # min 2
        embed_dropout_p=hparams["emb_dropout"],
        agg_type=hparams["agg"],
        agg_dim=hparams["agg_dim"],
        agg_n_layers=hparams["agg_depth"],  # min 2
        agg_dropout_p=hparams["agg_dropout"],
    )

    return system


def init_data_module(hparams: dict, cpus: int, dir_prefix: str) -> GraphDataModule:
    """Initalizes a Data Module for use during training, validation, and testing

    Args:
        hparams (dict): dictionary of hyperparameters
        cpus (int): number of cpus to use for parallel data loading
        dir_prefix (str): path prefix, used by ray tune mostly

    Returns:
        GraphDataModule: a configured data module
    """
    data_path = Path(dir_prefix, f"data/k{hparams['k']}/{hparams['datasetname']}")
    transforms = [
        SupportToAnomScore(hparams["C"]),
        Standardizer(),
        ThresholdAtDelta(hparams["DELTA"]),
    ]

    data_module = GraphDataModule(
        data_dir=data_path, transforms=transforms, num_workers=cpus
    )

    return data_module
