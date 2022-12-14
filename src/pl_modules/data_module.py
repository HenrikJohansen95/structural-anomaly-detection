from itertools import repeat
import os
import plotly.express as px

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from torch.utils.data import DataLoader


def graph_ds_collate(batch):
    g = batch[0][0]
    K = batch[0][1]
    y = batch[0][2]
    return (g, K, y)


class GraphDataset:
    """
    Loads and stores data as a list of <graph, patterns, support> tuples
    """

    def __init__(self, data_dir: str, kind: str) -> None:
        """Initializes the graph dataset

        Args:
            data_dir (str): parent directory of data
            kind (str): which data split this is
        """
        assert kind in ("test", "train", "validate")

        data_path = os.path.join(data_dir, f"{kind}.pt")

        # Data is stored in a list of tuples of the form:
        # <graph, list of patterns, list of support values>.
        self.data_list = torch.load(data_path)
        self.data_list = [list(e) for e in self.data_list]

        self.n_features = self.data_list[0][0].x.size(-1)
        self.e_features = self.data_list[0][0].edge_attr.size(-1)
        self.k = self.data_list[0][1].size(-1)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def to(self, device):
        for sample in self.data_list:
            for i in range(len(sample)):
                sample[i] = sample[i].to(device)
        return self


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, transforms, num_workers: int = 8):
        """Initializes graph data module for use with pytorch lightning

        Args:
            data_dir (str): directory where generated data is stored
            transforms (list): list of functions that are applied to each dataset
            num_workers (int, optional): workers to load data in parallel. Defaults to 8.
        """
        super().__init__()
        self.data_path = data_dir
        self.num_workers = num_workers
        self.transforms = transforms

    def setup(self, stage=None) -> None:
        # All datasets have to be initiated at the same time for the transform
        # metrics to be calculated correctly.
        self.data_train = GraphDataset(self.data_path, "train")
        for transform in self.transforms:
            transform(self.data_train, training=True)

        self.data_val = GraphDataset(self.data_path, "validate")
        for transform in self.transforms:
            transform(self.data_val)

        self.data_test = GraphDataset(self.data_path, "test")
        for transform in self.transforms:
            transform(self.data_test)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            collate_fn=graph_ds_collate,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
            persistent_workers=self.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            collate_fn=graph_ds_collate,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            collate_fn=graph_ds_collate,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0,
            persistent_workers=self.num_workers > 0,
        )

    def save_distribution_plot(
        self,
        datasetname,
        bins=16,
        binwidth=None,
        dir="./",
        dataset=None,
        unique=True,
        rug=True,
    ):
        test_anoms = torch.cat([e[2] for e in self.data_test.data_list])
        fit_anoms = torch.cat([e[2] for e in self.data_train.data_list])
        validate_anoms = torch.cat([e[2] for e in self.data_val.data_list])

        x = []
        ds = []
        if dataset == "test" or dataset is None:
            x.extend(test_anoms.flatten().tolist())
            ds.extend(repeat("test", test_anoms.size(0)))
        if dataset == "train" or dataset is None:
            x.extend(fit_anoms.flatten().tolist())
            ds.extend(repeat("train", fit_anoms.size(0)))
        if dataset == "val" or dataset is None:
            x.extend(validate_anoms.flatten().tolist())
            ds.extend(repeat("val", validate_anoms.size(0)))

        df = pd.DataFrame.from_dict({"anom": x, "dataset": ds})
        df["anom"] = df["anom"].round(6)
        ax = sns.histplot(
            x="anom",
            data=df,
            bins=bins,
            binwidth=binwidth,
            stat="proportion",
            common_norm=False,
            hue="dataset",
            multiple="layer",
        )
        if rug:
            ax = sns.rugplot(data=df, x="anom", ax=ax)
        ax.set_title(datasetname + " distribution")
        if bins == 2:
            ax.set_xlim(0, 1)

        path = os.path.join(dir, datasetname)
        ax.get_figure().savefig(os.path.join(path, "distribution"))
        plt.clf()

        if unique:
            ax = sns.countplot(
                x="anom",
                data=df,
            )

            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            for ind, label in enumerate(ax.get_xticklabels()):
                if ind % 32 == 0:  # every 10th label is kept
                    label.set_visible(True)

                else:
                    label.set_visible(False)

            ax.set_title(datasetname + " unique values")
            ax.get_figure().savefig(os.path.join(path, "unique.png"))
            plt.clf()
