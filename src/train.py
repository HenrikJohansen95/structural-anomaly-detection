import typer
import yaml


def run(
    hparams_yml_path: str = typer.Argument(
        None, help="path to a yaml config with model and training hyperparameters"
    ),
    cpus: int = typer.Option(12, help="number of cpu cores to use"),
    gpus: int = typer.Option(1, help="number of gpus to use"),
    seed: int = typer.Option(42, help="seed for reproducability"),
    epochs: int = typer.Option(
        8, help="number of epochs (pass over the entire dataset) to train for"
    ),
):
    # Lazy imports in case of improper CLI arguments
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        LearningRateMonitor,
        ModelCheckpoint,
        RichModelSummary,
        RichProgressBar,
    )

    from init import init_data_module, init_system

    with open(hparams_yml_path) as f:
        hparams = yaml.safe_load(f)

    pl.seed_everything(seed, workers=True)
    system = init_system(hparams, ".")
    data_module = init_data_module(hparams, cpus, ".")

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=gpus,
        precision=16,
        callbacks=[
            RichProgressBar(),
            RichModelSummary(2),
            LearningRateMonitor(),
            ModelCheckpoint(save_top_k=1, monitor="validation_mcc", mode="max"),
        ],
    )

    trainer.fit(system, datamodule=data_module)
    trainer.test(system, datamodule=data_module)


if __name__ == "__main__":
    typer.run(run)
