from pathlib import Path

import pytorch_lightning as pl
import ray
import typer
import yaml
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from init import init_data_module, init_system
from pl_modules.data_module import GraphDataModule
from pl_modules.system import GraphSystem
from pl_modules.transforms import Standardizer, SupportToAnomScore, ThresholdAtDelta
from enum import Enum

# k - dataset - delta combinations
SETS = [
    (2, "imdb", [1.1, 1.2]),
    (2, "lubm", [0.6, 0.7]),
    (2, "lastfm", [0.4, 0.7]),
    (3, "imdb", [0.6, 0.625]),
    (3, "lubm", [0.425, 0.5]),
    (3, "lastfm", [0.6, 0.8]),
]
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Aggregator(str, Enum):
    ffn = "ffn"
    mean = "mean"


class Embeddor(str, Enum):
    graphsage = "gs"
    gcn = "gcn"
    rgcn = "rgcn"


def trainable(hparams, cpus, gpus, seed, epochs, dir_prefix, checkpoint_dir=None):
    pl.seed_everything(seed, workers=True)
    system = init_system(hparams, dir_prefix)
    data_module = init_data_module(hparams, cpus, dir_prefix)

    trainer = pl.Trainer(
        max_epochs=epochs,
        precision=16,
        gpus=gpus,
        resume_from_checkpoint=Path(checkpoint_dir, "checkpoint")
        if checkpoint_dir
        else None,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "ap": "validation_ap",
                    "train_ap": "training_ap",
                    "roc": "validation_roc",
                    "f1": "validation_f1",
                    "pr": "validation_pr",
                    "rc": "validation_rc",
                    "mcc": "validation_mcc",
                    "tp": "validation_tp",
                    "fp": "validation_fp",
                    "tn": "validation_tn",
                    "fn": "validation_fn",
                },
                filename="checkpoint",
                on="validation_end",
            ),
        ],
    )

    trainer.fit(system, datamodule=data_module)


def test_model(
    best_result,
):
    k = best_result.config["k"]
    data_path = Path("./", f"data/k{k}/{best_result.config['datasetname']}")

    transforms = [
        SupportToAnomScore(best_result.config["C"]),
        Standardizer(),
        ThresholdAtDelta(best_result.config["DELTA"]),
    ]

    data_module = GraphDataModule(
        data_dir=data_path, transforms=transforms, num_workers=0
    )

    best_model = GraphSystem.load_from_checkpoint(
        Path(best_result.checkpoint.to_directory(), "checkpoint")
    )
    best_model.eval()

    trainer = pl.Trainer(
        precision=16,
        gpus=1,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    test_result = trainer.test(best_model, datamodule=data_module)
    return test_result[0]


def run(
    k: int = typer.Argument(..., help="size of patterns in dataset"),
    datasetname: str = typer.Argument(..., help="dataset"),
    delta: float = typer.Argument(..., help="negative/positive threshold"),
    emb: Embeddor = typer.Argument(..., help="embedding function"),
    agg: Aggregator = typer.Argument(..., help="aggregation function"),
    cpus: int = typer.Option(12, help="number of threads to use"),
    gpus: int = typer.Option(1, help="number of available gpus"),
    epochs: int = typer.Option(8, help="number of epochs to train for"),
    hparam_search_samples: int = typer.Option(
        16, help="how many hyperparameter sets to sample"
    ),
):
    ray.init(num_cpus=cpus, num_gpus=gpus)

    hparam_space = {
        "datasetname": datasetname,
        "DELTA": delta,
        "C": 1e9,
        "k": k,
        "lr": tune.loguniform(5e-6, 5e-3),
        "wd": 0.2,
        "pos_weight": tune.uniform(0.1, 10),
        "emb": emb,
        "emb_dim": tune.lograndint(8, 1024),
        "emb_depth": tune.randint(2, 10),
        "emb_dropout": 0.2,
    }

    if agg == "mean":
        hparam_space.update(
            {
                "agg": "mean",
                "agg_dim": None,
                "agg_depth": None,
                "agg_dropout": None,
            }
        )
    elif agg == "ffn":
        hparam_space.update(
            {
                "agg": "ffn",
                "agg_dim": tune.lograndint(8, 1024),
                "agg_depth": tune.randint(2, 10),
                "agg_dropout": 0.4,
            }
        )

    scheduler = ASHAScheduler(max_t=epochs, reduction_factor=2, grace_period=1)
    search = HyperOptSearch(
        n_initial_points=4,
    )

    # Multiple workers caused race condition issues on my computer when using
    # ray tune.
    train_with_params = tune.with_parameters(
        trainable, num_workers=0, seed=42, epochs=epochs, dir_prefix="../../../"
    )
    reporter = tune.CLIReporter(
        sort_by_metric=True,
        max_progress_rows=16,
        infer_limit=10,
        parameter_columns=[
            key
            for key, value in hparam_space.items()
            if isinstance(value, tune.search.sample.Domain)
        ],
    )
    tuner = tune.Tuner(
        tune.with_resources(
            train_with_params,
            resources={"cpu": cpus, "gpu": gpus},
        ),
        tune_config=tune.TuneConfig(
            metric="mcc",
            mode="max",
            scheduler=scheduler,
            search_alg=search,
            num_samples=hparam_search_samples,
            reuse_actors=False,
        ),
        param_space=hparam_space,
        run_config=air.RunConfig(
            local_dir="./runs",
            progress_reporter=reporter,
            verbose=1,
            checkpoint_config=air.CheckpointConfig(
                1,
            ),
        ),
    )

    results = tuner.fit()

    path = f"plots/k{hparam_space['k']}/{hparam_space['datasetname']}/delta{round(hparam_space['DELTA'], 4)}/{hparam_space['agg']}/{hparam_space['emb']}"
    Path(path).mkdir(parents=True, exist_ok=True)

    df = results.get_dataframe()
    df.to_csv(Path(path, "out.csv"))

    best_result = results.get_best_result()
    best_config = best_result.config
    with open(Path(path, "best_config.yml"), "w") as f:
        yaml.safe_dump(best_config, f)

    test_results = test_model(best_result)
    with open(Path(path, "test_results.yml"), "w") as f:
        yaml.safe_dump(test_results, f)


if __name__ == "__main__":
    typer.run(run)
