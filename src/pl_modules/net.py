from typing import Optional, Union
import torch
import torch_geometric.nn as t_gnn
from torch import nn
from torch.nn import functional as F


class BaseRelationalModel(nn.Module):
    """
    Based entirely on the design of t_gnn.models.BasicGNN
    """

    def __init__(
        self,
        in_channels: int,
        num_relations: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(
            self.init_conv(
                in_channels,
                hidden_channels,
                num_relations,
                num_bases,
                num_blocks,
                **kwargs,
            )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(
                    hidden_channels,
                    hidden_channels,
                    num_relations,
                    num_bases,
                    num_blocks,
                    **kwargs,
                )
            )

        self.convs.append(
            self.init_conv(
                hidden_channels,
                hidden_channels,
                num_relations,
                num_bases,
                num_blocks,
                **kwargs,
            )
        )

    def init_conv(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int],
        num_blocks: Optional[int],
        **kwargs,
    ):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_type):
        for i in range(self.num_layers):
            x: torch.Tensor = self.convs[i](x, edge_index, edge_type)
            if i == self.num_layers - 1:
                break
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class RGCNModel(BaseRelationalModel):
    """
    Based entirely on t_gnn.models.GCN
    """

    def init_conv(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int],
        num_blocks: Optional[int],
        **kwargs,
    ):
        return t_gnn.RGCNConv(
            in_channels,
            out_channels,
            num_relations,
            num_bases=num_bases,
            num_blocks=num_blocks,
            **kwargs,
        )


class RGATModel(BaseRelationalModel):
    """
    Based entirely on t_gnn.models.GAT
    """

    def init_conv(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int],
        num_blocks: Optional[int],
        **kwargs,
    ):
        if "heads" in kwargs:
            # out_channels ought to be divisible by heads
            assert out_channels % kwargs["heads"] == 0
        if "concat" not in kwargs or kwargs["concat"]:
            out_channels = out_channels // kwargs.get("heads", 1)
        return t_gnn.RGATConv(
            in_channels,
            out_channels,
            num_relations,
            num_bases=num_bases,
            num_blocks=num_blocks,
            **kwargs,
        )


class LinearClassifierHead(nn.Module):
    def __init__(self, in_features, dim, num_layers, dropout_p) -> None:
        super().__init__()

        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(in_features, dim))
        for _ in range(num_layers - 2):
            self.lin_layers.append(nn.Linear(dim, dim))
        self.lin_layers.append(nn.Linear(dim, 1))

        self.num_layers = num_layers
        self.dropout_p = dropout_p

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.lin_layers[i](x)
            if i == self.num_layers - 1:
                break
            x = x.relu()
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x
