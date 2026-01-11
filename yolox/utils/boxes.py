#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision

__all__ = [
    "filter_box",
    "postprocess",
    "postprocess_polygon",
    "bboxes_iou",
    "polygon_iou",
    "polygon_iou_batch",
    "polygon_area",
    "polygon_centroid",
    "polygon_nms",
    "matrix_iou",
    "adjust_box_anns",
    "adjust_polygon_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def cxcywh2xyxy(bboxes):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes


# ===================== Polygon Bounding Box Utilities =====================


def polygon_area(polygon):
    """
    Compute the area of a polygon using the Shoelace formula.
    
    Args:
        polygon: Tensor of shape (N, 8) representing N quadrilaterals with 
                 4 corner points each: (x1,y1, x2,y2, x3,y3, x4,y4)
                 Points should be ordered (clockwise or counter-clockwise).
    
    Returns:
        Tensor of shape (N,) containing the area of each polygon.
    """
    if polygon.dim() == 1:
        polygon = polygon.unsqueeze(0)
    
    # Reshape to (N, 4, 2) - 4 points with x,y coordinates
    pts = polygon.view(-1, 4, 2)
    
    # Shoelace formula: 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    x = pts[:, :, 0]  # (N, 4)
    y = pts[:, :, 1]  # (N, 4)
    
    # Roll to get next points
    x_next = torch.roll(x, -1, dims=1)
    y_next = torch.roll(y, -1, dims=1)
    
    area = 0.5 * torch.abs(torch.sum(x * y_next - x_next * y, dim=1))
    return area


def polygon_area_np(polygon):
    """
    Numpy version of polygon_area for data augmentation.
    
    Args:
        polygon: ndarray of shape (N, 8) or (8,)
    
    Returns:
        ndarray of shape (N,) or scalar containing the area of each polygon.
    """
    if polygon.ndim == 1:
        polygon = polygon.reshape(1, -1)
    
    pts = polygon.reshape(-1, 4, 2)
    x = pts[:, :, 0]
    y = pts[:, :, 1]
    
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    
    area = 0.5 * np.abs(np.sum(x * y_next - x_next * y, axis=1))
    return area.squeeze() if polygon.shape[0] == 1 else area


def polygon_centroid(polygon):
    """
    Compute the centroid of polygons.
    
    Args:
        polygon: Tensor of shape (N, 8) - 4 corner points per polygon
    
    Returns:
        Tensor of shape (N, 2) containing (cx, cy) for each polygon.
    """
    if polygon.dim() == 1:
        polygon = polygon.unsqueeze(0)
    
    pts = polygon.view(-1, 4, 2)
    centroid = pts.mean(dim=1)  # Average of all 4 corners
    return centroid


def polygon_centroid_np(polygon):
    """
    Numpy version of polygon_centroid.
    
    Args:
        polygon: ndarray of shape (N, 8) or (8,)
    
    Returns:
        ndarray of shape (N, 2) or (2,) containing (cx, cy).
    """
    if polygon.ndim == 1:
        polygon = polygon.reshape(1, -1)
    
    pts = polygon.reshape(-1, 4, 2)
    centroid = pts.mean(axis=1)
    return centroid.squeeze() if polygon.shape[0] == 1 else centroid


def _line_intersection(p1, p2, p3, p4):
    """
    Compute intersection point of two line segments (p1-p2) and (p3-p4).
    Returns None if lines don't intersect or are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return None


def _point_in_polygon(point, polygon):
    """
    Check if a point is inside a convex polygon using cross product method.
    polygon: list of (x, y) tuples in order
    """
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)
        if cross < 0:
            return False
    return True


def _clip_polygon_sutherland_hodgman(subject, clip):
    """
    Sutherland-Hodgman polygon clipping algorithm.
    Clips the subject polygon against the clip polygon.
    Both polygons should be convex and vertices in counter-clockwise order.
    
    Args:
        subject: List of (x, y) tuples for the subject polygon
        clip: List of (x, y) tuples for the clipping polygon
    
    Returns:
        List of (x, y) tuples for the clipped polygon
    """
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


def _polygon_area_from_points(points):
    """Compute area of polygon from list of (x, y) points using Shoelace."""
    if len(points) < 3:
        return 0.0
    
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def polygon_iou(poly_a, poly_b):
    """
    Compute IoU between two quadrilaterals.
    
    Args:
        poly_a: Tensor of shape (8,) - single polygon with 4 corner points
        poly_b: Tensor of shape (8,) - single polygon with 4 corner points
    
    Returns:
        IoU value (float)
    """
    # Convert to list of (x, y) tuples
    pts_a = [(poly_a[i].item(), poly_a[i+1].item()) for i in range(0, 8, 2)]
    pts_b = [(poly_b[i].item(), poly_b[i+1].item()) for i in range(0, 8, 2)]
    
    # Compute intersection using Sutherland-Hodgman
    intersection_pts = _clip_polygon_sutherland_hodgman(pts_a, pts_b)
    
    if len(intersection_pts) < 3:
        return 0.0
    
    inter_area = _polygon_area_from_points(intersection_pts)
    area_a = _polygon_area_from_points(pts_a)
    area_b = _polygon_area_from_points(pts_b)
    
    union_area = area_a + area_b - inter_area
    
    if union_area < 1e-10:
        return 0.0
    
    return inter_area / union_area


def polygon_iou_batch(poly_a, poly_b):
    """
    Compute pairwise IoU between two sets of polygons.
    
    Args:
        poly_a: Tensor of shape (N, 8) - N polygons with 4 corner points each
        poly_b: Tensor of shape (M, 8) - M polygons with 4 corner points each
    
    Returns:
        Tensor of shape (N, M) containing pairwise IoU values.
    """
    N = poly_a.shape[0]
    M = poly_b.shape[0]
    
    ious = torch.zeros(N, M, device=poly_a.device, dtype=poly_a.dtype)
    
    for i in range(N):
        for j in range(M):
            ious[i, j] = polygon_iou(poly_a[i], poly_b[j])
    
    return ious


def polygon_nms(polygons, scores, iou_threshold):
    """
    Non-maximum suppression for polygonal bounding boxes.
    
    Args:
        polygons: Tensor of shape (N, 8) - polygon coordinates
        scores: Tensor of shape (N,) - confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Tensor of indices to keep after NMS.
    """
    if polygons.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=polygons.device)
    
    # Sort by score in descending order
    order = scores.argsort(descending=True)
    
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        
        if order.numel() == 1:
            break
        
        # Compute IoU with remaining boxes
        remaining = order[1:]
        ious = torch.zeros(remaining.numel(), device=polygons.device)
        
        for idx, j in enumerate(remaining):
            ious[idx] = polygon_iou(polygons[i], polygons[j.item()])
        
        # Keep boxes with IoU below threshold
        mask = ious <= iou_threshold
        order = remaining[mask]
    
    return torch.tensor(keep, dtype=torch.long, device=polygons.device)


def postprocess_polygon(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    """
    Postprocessing for polygon predictions.
    
    Args:
        prediction: Tensor of shape (batch, n_anchors, 9 + num_classes)
                   where 9 = 8 polygon coords + 1 objectness
        num_classes: Number of classes
        conf_thre: Confidence threshold
        nms_thre: IoU threshold for NMS
        class_agnostic: Whether to perform class-agnostic NMS
    
    Returns:
        List of detections per image, each with shape (n_det, 10)
        Format: (x1,y1, x2,y2, x3,y3, x4,y4, obj_conf, class_conf, class_pred)
    """
    output = [None for _ in range(len(prediction))]
    
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue
        
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 9: 9 + num_classes], 1, keepdim=True
        )
        
        conf_mask = (image_pred[:, 8] * class_conf.squeeze() >= conf_thre).squeeze()
        
        # Detections: (8 polygon coords, obj_conf, class_conf, class_pred)
        detections = torch.cat((
            image_pred[:, :9],  # 8 coords + obj_conf
            class_conf,
            class_pred.float()
        ), 1)
        detections = detections[conf_mask]
        
        if not detections.size(0):
            continue
        
        if class_agnostic:
            nms_out_index = polygon_nms(
                detections[:, :8],
                detections[:, 8] * detections[:, 9],
                nms_thre,
            )
        else:
            # Class-aware NMS: process each class separately
            unique_classes = detections[:, 10].unique()
            nms_out_index = []
            
            for cls in unique_classes:
                cls_mask = detections[:, 10] == cls
                cls_dets = detections[cls_mask]
                cls_indices = torch.where(cls_mask)[0]
                
                cls_nms = polygon_nms(
                    cls_dets[:, :8],
                    cls_dets[:, 8] * cls_dets[:, 9],
                    nms_thre,
                )
                nms_out_index.extend(cls_indices[cls_nms].tolist())
            
            nms_out_index = torch.tensor(nms_out_index, device=detections.device)
        
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    
    return output


def adjust_polygon_anns(polygon, scale_ratio, padw, padh, w_max, h_max):
    """
    Adjust polygon annotations for data augmentation.
    
    Args:
        polygon: ndarray of shape (N, 8) - polygon coordinates
        scale_ratio: Scale factor
        padw: Padding width offset
        padh: Padding height offset
        w_max: Maximum width for clipping
        h_max: Maximum height for clipping
    
    Returns:
        Adjusted polygon coordinates.
    """
    # Scale and pad all coordinates
    polygon[:, 0::2] = np.clip(polygon[:, 0::2] * scale_ratio + padw, 0, w_max)
    polygon[:, 1::2] = np.clip(polygon[:, 1::2] * scale_ratio + padh, 0, h_max)
    return polygon


def filter_polygon(output, scale_range):
    """
    Filter polygons by area.
    
    Args:
        output: Tensor of shape (N, 9+class) - polygons with 8 coords
        scale_range: (min_scale, max_scale) tuple
    
    Returns:
        Filtered output tensor.
    """
    min_scale, max_scale = scale_range
    areas = polygon_area(output[:, :8])
    keep = (areas > min_scale * min_scale) & (areas < max_scale * max_scale)
    return output[keep]

