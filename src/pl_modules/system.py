import pytorch_lightning as pl
import torch
import torch_geometric.nn
from torch.nn import functional as F
from torchmetrics import functional as tmf

from pl_modules.net import LinearClassifierHead, RGCNModel


class GraphSystem(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        wd: float,
        pos_sample_weight: float,
        node_features: int,
        edge_features: int,
        k: int,
        embed_type: str,
        embed_dim: int,
        embed_n_layers: int,
        embed_dropout_p: float,
        agg_type: str,
        agg_dim: int,
        agg_n_layers: int,
        agg_dropout_p: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert embed_type in ("gcn", "rgcn", "gs")
        assert agg_type in ("mean", "ffn")

        # Setup Model
        if embed_type == "gcn":
            self.embedder = torch_geometric.nn.models.GCN(
                in_channels=node_features,
                hidden_channels=embed_dim,
                num_layers=embed_n_layers,
                dropout=embed_dropout_p,
            )
        elif embed_type == "rgcn":
            self.embedder = RGCNModel(
                in_channels=node_features,
                num_relations=edge_features,
                hidden_channels=embed_dim,
                num_layers=embed_n_layers,
                dropout=embed_dropout_p,
            )
        elif embed_type == "gs":
            self.embedder = torch_geometric.nn.models.GraphSAGE(
                in_channels=node_features,
                hidden_channels=embed_dim,
                num_layers=embed_n_layers,
                dropout=embed_dropout_p,
            )

        if agg_type == "ffn":
            # if embed_type in ("rgat", "gat"):
            #     lin_in_features = embed_dim * k * n_heads
            # else:
            lin_in_features = embed_dim * k
            self.aggregator = LinearClassifierHead(
                lin_in_features, agg_dim, agg_n_layers, agg_dropout_p
            )

        elif agg_type == "mean":
            self.aggregator = lambda nodes: torch.mean(nodes, -1, keepdim=True)

        self.embed_type = embed_type

    def forward(self, g, node_indices):
        embedding = self._embed(g)
        nodes = embedding[node_indices]
        nodes = nodes.flatten(1)
        y_hat = self.aggregator(nodes)
        return y_hat

    def _embed(self, g):
        if self.embed_type in ("rgcn"):
            embedding = self.embedder(g.x, g.edge_index, g.edge_attr.argmax(-1))
        else:
            embedding = self.embedder(g.x, g.edge_index)
        return embedding

    def training_step(self, batch, batch_idx):
        g_i, K_i_indices, y_i = batch

        embedding = self._embed(g_i)
        nodes = embedding[K_i_indices]  # [batch, k, features]
        nodes = nodes.flatten(1)  # [batch, k * features]
        y_hat = self.aggregator(nodes)

        loss = F.binary_cross_entropy_with_logits(
            y_hat, y_i, pos_weight=torch.tensor(self.hparams.pos_sample_weight)
        )

        t_ap = float(tmf.average_precision(y_hat, y_i, task="binary"))
        return {"loss": loss, "ap": t_ap}

    def training_epoch_end(self, outputs):
        loss = float(torch.tensor([float(e["loss"]) for e in outputs]).nanmean())
        ap = float(torch.tensor([float(e["ap"]) for e in outputs]).nanmean())
        self.log("training_loss", loss)
        self.log("training_ap", ap)

    def validation_step(self, batch, batch_idx):
        g_i, K_i_indices, y_i = batch

        embedding = self._embed(g_i)
        nodes = embedding[K_i_indices]
        nodes = nodes.flatten(1)  # [batch, k * features]
        y_hat = self.aggregator(nodes).sigmoid()

        return {"ground_truth": y_i, "preds": y_hat}

    def validation_epoch_end(self, outputs):
        ground_truth = torch.cat([e["ground_truth"] for e in outputs])
        preds = torch.cat([e["preds"] for e in outputs])

        self.log_all(preds, ground_truth, title="validation")

    def test_step(self, batch, batch_idx):
        g_i, K_i_indices, y_i = batch

        embedding = self._embed(g_i)
        nodes = embedding[K_i_indices]
        nodes = nodes.flatten(1)  # [batch, k * features]
        y_hat = self.aggregator(nodes).sigmoid()

        return {"ground_truth": y_i, "preds": y_hat}

    def test_epoch_end(self, outputs):
        ground_truth = torch.cat([e["ground_truth"] for e in outputs])
        preds = torch.cat([e["preds"] for e in outputs])

        pos_ind = (ground_truth == 1).flatten().nonzero()
        neg_ind = (ground_truth == 0).flatten().nonzero()

        print("pos mean:", float(preds[pos_ind].mean()))
        print("pos std:", float(preds[pos_ind].std()))
        print("neg mean:", float(preds[neg_ind].mean()))
        print("neg std:", float(preds[neg_ind].std()))

        self.log_all(preds, ground_truth, title="test")

    def log_all(self, preds: torch.Tensor, ground_truth: torch.Tensor, title: str):
        """Logs performance metrics

        Args:
            preds (torch.Tensor): model predictions
            ground_truth (torch.Tensor): ground truth / labels
            title (str): title or prefix for stat (i.e. train / val / test)
        """
        digits = 3

        ap = tmf.average_precision(preds, ground_truth.short(), task="binary")
        self.log(f"{title}_ap", round(float(ap), digits))

        roc = tmf.auroc(preds, ground_truth.short(), task="binary")
        self.log(f"{title}_roc", round(float(roc), digits))

        pr = tmf.precision(preds, ground_truth.short(), task="binary")
        self.log(f"{title}_pr", round(float(pr), digits))

        rc = tmf.recall(preds, ground_truth.short(), task="binary")
        self.log(f"{title}_rc", round(float(rc), digits))

        f1 = tmf.f1_score(preds, ground_truth.short(), task="binary")
        self.log(f"{title}_f1", round(float(f1), digits))

        mcc = tmf.matthews_corrcoef(preds, ground_truth.short(), task="binary")
        self.log(f"{title}_mcc", round(float(mcc), digits))

        cm = tmf.confusion_matrix(
            preds, ground_truth.short(), num_classes=2, normalize="true", task="binary"
        )
        [tn, fp], [fn, tp] = cm
        self.log(f"{title}_tn", round(float(tn), digits))
        self.log(f"{title}_fn", round(float(fn), digits))
        self.log(f"{title}_fp", round(float(fp), digits))
        self.log(f"{title}_tp", round(float(tp), digits))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), self.hparams.lr, weight_decay=self.hparams.wd
        )
        lr_s_config = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer=opt,
                max_lr=self.hparams.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                div_factor=3,
                final_div_factor=20,
            ),
            "interval": "step",
        }

        return {"optimizer": opt, "lr_scheduler": lr_s_config}
