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
    """
    IoU loss for 4-point polygon bounding boxes.
    Uses Sutherland-Hodgman algorithm for polygon intersection.

    The loss formula is `1 - IoU^2` (squared IoU), which is consistent with
    the standard YOLOX IOUloss implementation. The squared variant provides
    stronger gradients for high IoU values, encouraging more precise localization.
    """

    def __init__(self, reduction="none"):
        super(PolygonIOULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Compute polygon IoU loss using the formula: loss = 1 - IoU^2

        The squared IoU loss provides stronger gradients for high IoU predictions,
        which helps achieve more precise localization.

        Args:
            pred: tensor of shape (N, 8) - predicted polygons (x1,y1,x2,y2,x3,y3,x4,y4)
            target: tensor of shape (N, 8) - target polygons

        Returns:
            Loss tensor
        """
        assert pred.shape[0] == target.shape[0]
        assert pred.shape[1] == 8 and target.shape[1] == 8

        n = pred.shape[0]
        if n == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        losses = torch.zeros(n, device=pred.device, dtype=pred.dtype)
        for i in range(n):
            iou = polygon_iou(pred[i], target[i])
            losses[i] = 1 - iou ** 2

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        return losses
