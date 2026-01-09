#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou", use_oriented_bbox=False):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type
        self.use_oriented_bbox = use_oriented_bbox

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        if self.use_oriented_bbox:
            # For oriented bounding boxes, we compute IoU using only cx, cy, w, h
            # The angle is handled separately in L1 loss
            pred = pred.view(-1, 5)
            target = target.view(-1, 5)
            pred_box = pred[:, :4]
            target_box = target[:, :4]
        else:
            pred_box = pred.view(-1, 4)
            target_box = target.view(-1, 4)
        
        tl = torch.max(
            (pred_box[:, :2] - pred_box[:, 2:] / 2), (target_box[:, :2] - target_box[:, 2:] / 2)
        )
        br = torch.min(
            (pred_box[:, :2] + pred_box[:, 2:] / 2), (target_box[:, :2] + target_box[:, 2:] / 2)
        )

        area_p = torch.prod(pred_box[:, 2:], 1)
        area_g = torch.prod(target_box[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred_box[:, :2] - pred_box[:, 2:] / 2), (target_box[:, :2] - target_box[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred_box[:, :2] + pred_box[:, 2:] / 2), (target_box[:, :2] + target_box[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
