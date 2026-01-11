#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


def polygon_area(coords):
    """
    Calculate area of polygon using Shoelace formula.
    coords: tensor of shape (N, 8) representing N polygons with 4 points each
    Returns: tensor of shape (N,) with areas
    """
    # Extract x and y coordinates
    x1, y1 = coords[:, 0], coords[:, 1]
    x2, y2 = coords[:, 2], coords[:, 3]
    x3, y3 = coords[:, 4], coords[:, 5]
    x4, y4 = coords[:, 6], coords[:, 7]
    
    # Shoelace formula for quadrilateral
    area = 0.5 * torch.abs(
        x1 * y2 - x2 * y1 +
        x2 * y3 - x3 * y2 +
        x3 * y4 - x4 * y3 +
        x4 * y1 - x1 * y4
    )
    return area


def polygon_to_bbox(coords):
    """
    Convert polygon coordinates to bounding box (min_x, min_y, max_x, max_y).
    coords: tensor of shape (N, 8)
    Returns: tensor of shape (N, 4)
    """
    x_coords = coords[:, [0, 2, 4, 6]]
    y_coords = coords[:, [1, 3, 5, 7]]
    
    min_x = x_coords.min(dim=1)[0]
    max_x = x_coords.max(dim=1)[0]
    min_y = y_coords.min(dim=1)[0]
    max_y = y_coords.max(dim=1)[0]
    
    return torch.stack([min_x, min_y, max_x, max_y], dim=1)


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        # Check if polygon format (8 coordinates) or bbox format (4 coordinates)
        if pred.shape[1] == 8:
            # Polygon format - use bounding box approximation for IoU
            pred = pred.view(-1, 8)
            target = target.view(-1, 8)
            
            # Convert polygons to bounding boxes for IoU calculation
            pred_bbox = polygon_to_bbox(pred)
            target_bbox = polygon_to_bbox(target)
            
            # Calculate IoU using bounding boxes
            tl = torch.max(pred_bbox[:, :2], target_bbox[:, :2])
            br = torch.min(pred_bbox[:, 2:], target_bbox[:, 2:])
            
            area_p = (pred_bbox[:, 2] - pred_bbox[:, 0]) * (pred_bbox[:, 3] - pred_bbox[:, 1])
            area_g = (target_bbox[:, 2] - target_bbox[:, 0]) * (target_bbox[:, 3] - target_bbox[:, 1])
            
            en = (tl < br).type(tl.type()).prod(dim=1)
            area_i = torch.prod(br - tl, 1) * en
            area_u = area_p + area_g - area_i
            iou = (area_i) / (area_u + 1e-16)
            
            if self.loss_type == "iou":
                loss = 1 - iou ** 2
            elif self.loss_type == "giou":
                c_tl = torch.min(pred_bbox[:, :2], target_bbox[:, :2])
                c_br = torch.max(pred_bbox[:, 2:], target_bbox[:, 2:])
                area_c = torch.prod(c_br - c_tl, 1)
                giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
                loss = 1 - giou.clamp(min=-1.0, max=1.0)
        else:
            # Original bbox format (4 coordinates: cx, cy, w, h)
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
