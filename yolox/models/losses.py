#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


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
    IoU Loss for polygon bounding boxes (4 corner points = 8 coordinates).
    
    Uses Sutherland-Hodgman algorithm for polygon intersection computation.
    """
    
    def __init__(self, reduction="none", loss_type="iou"):
        super(PolygonIOULoss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        Compute polygon IoU loss.
        
        Args:
            pred: Tensor of shape (N, 8) - predicted polygon coordinates
            target: Tensor of shape (N, 8) - target polygon coordinates
            Each polygon is (x1,y1, x2,y2, x3,y3, x4,y4)
        
        Returns:
            IoU loss values.
        """
        assert pred.shape[0] == target.shape[0]
        
        pred = pred.view(-1, 8)
        target = target.view(-1, 8)
        
        n = pred.shape[0]
        loss = torch.zeros(n, device=pred.device, dtype=pred.dtype)
        
        for i in range(n):
            iou = self._compute_polygon_iou(pred[i], target[i])
            if self.loss_type == "iou":
                loss[i] = 1 - iou ** 2
            elif self.loss_type == "giou":
                # For GIoU, compute convex hull area (simplified version)
                loss[i] = 1 - iou
            else:
                loss[i] = 1 - iou ** 2

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
    
    def _compute_polygon_iou(self, poly_a, poly_b):
        """Compute IoU between two quadrilaterals."""
        # Convert to list of (x, y) tuples
        pts_a = [(poly_a[i].item(), poly_a[i+1].item()) for i in range(0, 8, 2)]
        pts_b = [(poly_b[i].item(), poly_b[i+1].item()) for i in range(0, 8, 2)]
        
        # Compute intersection using Sutherland-Hodgman
        intersection_pts = self._clip_polygon(pts_a, pts_b)
        
        if len(intersection_pts) < 3:
            return torch.tensor(0.0, device=poly_a.device, dtype=poly_a.dtype)
        
        inter_area = self._polygon_area(intersection_pts)
        area_a = self._polygon_area(pts_a)
        area_b = self._polygon_area(pts_b)
        
        union_area = area_a + area_b - inter_area
        
        if union_area < 1e-10:
            return torch.tensor(0.0, device=poly_a.device, dtype=poly_a.dtype)
        
        iou = inter_area / union_area
        return torch.tensor(iou, device=poly_a.device, dtype=poly_a.dtype)
    
    def _clip_polygon(self, subject, clip):
        """Sutherland-Hodgman polygon clipping algorithm."""
        def inside(p, edge_start, edge_end):
            return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) > \
                   (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])
        
        def intersection(p1, p2, edge_start, edge_end):
            dc = (edge_start[0] - edge_end[0], edge_start[1] - edge_end[1])
            dp = (p1[0] - p2[0], p1[1] - p2[1])
            n1 = (edge_start[0] - p1[0], edge_start[1] - p1[1])
            
            denom = dc[0] * dp[1] - dc[1] * dp[0]
            if abs(denom) < 1e-10:
                return p1
            
            t = (n1[0] * dc[1] - n1[1] * dc[0]) / denom
            return (p1[0] + t * dp[0], p1[1] + t * dp[1])
        
        output = list(subject)
        
        for i in range(len(clip)):
            if len(output) == 0:
                return []
            
            input_list = output
            output = []
            
            edge_start = clip[i]
            edge_end = clip[(i + 1) % len(clip)]
            
            for j in range(len(input_list)):
                current = input_list[j]
                previous = input_list[j - 1]
                
                if inside(current, edge_start, edge_end):
                    if not inside(previous, edge_start, edge_end):
                        output.append(intersection(previous, current, edge_start, edge_end))
                    output.append(current)
                elif inside(previous, edge_start, edge_end):
                    output.append(intersection(previous, current, edge_start, edge_end))
        
        return output
    
    def _polygon_area(self, points):
        """Compute area of polygon using Shoelace formula."""
        if len(points) < 3:
            return 0.0
        
        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0

