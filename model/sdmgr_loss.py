# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.losses import accuracy


from mmocr.models.builder import LOSSES
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
class SDMGRLoss(nn.Cell):
    """The implementation the loss of key information extraction proposed in
    the paper: Spatial Dual-Modality Graph Reasoning for Key Information
    Extraction.

    https://arxiv.org/abs/2103.14470.
    """

    def __init__(self, node_weight=1.0, edge_weight=1.0, ignore=-100):
        super().__init__()
        self.loss_node = nn.CrossEntropyLoss(ignore_index=ignore)
        self.loss_edge = nn.CrossEntropyLoss(ignore_index=-1)
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.ignore = ignore

    def forward(self, node_preds, edge_preds, gts):
        node_gts, edge_gts = [], []
        for gt in gts:
            node_gts.append(gt[:, 0])
            edge_gts.append(gt[:, 1:].contiguous().view(-1))
        concat_op = ops.Concat()
        cast_op = ops.Cast()
        node_gts = concat_op(cast_op((node_gts).long()))
        edge_gts = concat_op(cast_op((edge_gts).long()))

        node_valids = torch.nonzero(
            node_gts != self.ignore, as_tuple=False).view(-1)
        edge_valids = torch.nonzero(edge_gts != -1, as_tuple=False).view(-1)
        return dict(
            loss_node=self.node_weight * self.loss_node(node_preds, node_gts),
            loss_edge=self.edge_weight * self.loss_edge(edge_preds, edge_gts))
