#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from yolox.utils import polygon_iou


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class PolygonIOULoss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(PolygonIOULoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        Calculate IoU loss for polygon predictions.
        
        Args:
            pred: (N, 8) tensor of predicted polygon vertices
            target: (N, 8) tensor of target polygon vertices
        
        Returns:
            loss: IoU loss values
        """
        assert pred.shape[0] == target.shape[0]
        assert pred.shape[1] == 8 and target.shape[1] == 8
        
        pred = pred.view(-1, 8)
        target = target.view(-1, 8)
        
        # Calculate IoU using polygon_iou function
        # For loss calculation, compute full IoU matrix and extract diagonal
        iou_matrix = polygon_iou(pred, target, use_torch=True)
        
        # Extract diagonal elements (pairwise IoU)
        ious = torch.diagonal(iou_matrix)
        
        if self.loss_type == "iou":
            loss = 1 - ious ** 2
        elif self.loss_type == "giou":
            # For polygons, GIoU is more complex, so we use standard IoU
            # TODO: Implement proper GIoU for polygons if needed
            loss = 1 - ious
        else:
            loss = 1 - ious

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
