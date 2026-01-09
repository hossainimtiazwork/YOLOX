#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

# Constants
POLYGON_COORDS = 8  # Number of values in polygon format (4 points with x,y each)


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, pred.shape[-1])
        target = target.view(-1, target.shape[-1])
        
        # Handle polygon format (8 values) - convert to bounding box for IoU calculation
        if pred.shape[1] == POLYGON_COORDS:
            # Convert polygon to center-based representation for IoU
            # Calculate bounding box from polygon points
            pred_x_coords = pred[:, [0, 2, 4, 6]]
            pred_y_coords = pred[:, [1, 3, 5, 7]]
            target_x_coords = target[:, [0, 2, 4, 6]]
            target_y_coords = target[:, [1, 3, 5, 7]]
            
            pred_x1 = torch.min(pred_x_coords, dim=1)[0]
            pred_y1 = torch.min(pred_y_coords, dim=1)[0]
            pred_x2 = torch.max(pred_x_coords, dim=1)[0]
            pred_y2 = torch.max(pred_y_coords, dim=1)[0]
            
            target_x1 = torch.min(target_x_coords, dim=1)[0]
            target_y1 = torch.min(target_y_coords, dim=1)[0]
            target_x2 = torch.max(target_x_coords, dim=1)[0]
            target_y2 = torch.max(target_y_coords, dim=1)[0]
            
            # Convert to center and size format for IoU calculation
            pred_cx = (pred_x1 + pred_x2) / 2
            pred_cy = (pred_y1 + pred_y2) / 2
            pred_w = pred_x2 - pred_x1
            pred_h = pred_y2 - pred_y1
            pred = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)
            
            target_cx = (target_x1 + target_x2) / 2
            target_cy = (target_y1 + target_y2) / 2
            target_w = target_x2 - target_x1
            target_h = target_y2 - target_y1
            target = torch.stack([target_cx, target_cy, target_w, target_h], dim=1)
        
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
