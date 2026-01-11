#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision

# Epsilon for floating point comparisons
POLYGON_EPS = 1e-10

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy",
    "polygon_area",
    "sutherland_hodgman_clip",
    "polygon_iou",
    "polygon_iou_batch",
    "polygon_nms",
    "polygon_nms",
    "postprocess_polygon",
    "order_vertices_torch",
]


def order_vertices_torch(polys):
    """
    polys: (N, 8) or (N, 4, 2)
    Returns ordered (N, 8)
    """
    # device = polys.device
    if polys.dim() == 2 and polys.shape[1] == 8:
        polys = polys.view(-1, 4, 2)

    # calc centroid
    center = torch.mean(polys, dim=1, keepdim=True) # (N, 1, 2)

    # calc angles
    diff = polys - center
    angles = torch.atan2(diff[..., 1], diff[..., 0]) # (N, 4)

    # sort
    _, indices = torch.sort(angles, dim=1) # (N, 4)

    # gather
    indices = indices.unsqueeze(-1).expand(-1, -1, 2) # (N, 4, 2)
    sorted_polys = torch.gather(polys, 1, indices)

    return sorted_polys.view(-1, 8)


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


# ==================== Polygon IoU Functions ====================


def polygon_area(polygon):
    """
    Compute the area of a polygon using the Shoelace formula.

    Args:
        polygon: tensor of shape (N, 2) representing N vertices of the polygon

    Returns:
        Absolute area of the polygon
    """
    n = polygon.shape[0]
    if n < 3:
        return torch.tensor(0.0, device=polygon.device, dtype=polygon.dtype)

    # Shoelace formula
    x = polygon[:, 0]
    y = polygon[:, 1]
    shifted_x = torch.roll(x, -1)
    shifted_y = torch.roll(y, -1)
    area = torch.abs(torch.sum(x * shifted_y - shifted_x * y)) / 2.0
    return area


def _line_intersection(p1, p2, p3, p4):
    """
    Compute the intersection point of two line segments (p1-p2) and (p3-p4).
    Used in Sutherland-Hodgman algorithm.

    Args:
        p1, p2: endpoints of first line segment (each of shape (2,))
        p3, p4: endpoints of second line segment (each of shape (2,))

    Returns:
        Intersection point of shape (2,)
    """
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if torch.abs(denom) < POLYGON_EPS:
        return p1  # Lines are parallel, return p1 as fallback

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return torch.stack([x, y])


def _is_inside_edge(point, edge_start, edge_end):
    """
    Check if a point is on the inside (left side) of an edge.

    Args:
        point: point to check (shape (2,))
        edge_start: start of edge (shape (2,))
        edge_end: end of edge (shape (2,))

    Returns:
        Boolean indicating if point is inside
    """
    return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) >= \
           (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])


def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    """
    Sutherland-Hodgman polygon clipping algorithm.
    Clips the subject polygon against the clip polygon.

    Args:
        subject_polygon: tensor of shape (N, 2) - polygon to be clipped
        clip_polygon: tensor of shape (M, 2) - clipping polygon (must be convex)

    Returns:
        Clipped polygon as tensor of shape (K, 2), may be empty
    """
    if subject_polygon.shape[0] < 3 or clip_polygon.shape[0] < 3:
        return torch.zeros((0, 2), device=subject_polygon.device, dtype=subject_polygon.dtype)

    output_list = [subject_polygon[i] for i in range(subject_polygon.shape[0])]
    num_clip_vertices = clip_polygon.shape[0]

    for i in range(num_clip_vertices):
        if len(output_list) == 0:
            break

        input_list = output_list
        output_list = []

        edge_start = clip_polygon[i]
        edge_end = clip_polygon[(i + 1) % num_clip_vertices]

        for j in range(len(input_list)):
            current = input_list[j]
            previous = input_list[j - 1]

            current_inside = _is_inside_edge(current, edge_start, edge_end)
            previous_inside = _is_inside_edge(previous, edge_start, edge_end)

            if current_inside:
                if not previous_inside:
                    # Entering: add intersection
                    intersection = _line_intersection(previous, current, edge_start, edge_end)
                    output_list.append(intersection)
                output_list.append(current)
            elif previous_inside:
                # Leaving: add intersection
                intersection = _line_intersection(previous, current, edge_start, edge_end)
                output_list.append(intersection)

    if len(output_list) == 0:
        return torch.zeros((0, 2), device=subject_polygon.device, dtype=subject_polygon.dtype)

    return torch.stack(output_list)


def polygon_iou(poly1, poly2):
    """
    Compute IoU between two polygons using Sutherland-Hodgman clipping.

    Args:
        poly1: tensor of shape (4, 2) or (8,) - first polygon (4 points)
        poly2: tensor of shape (4, 2) or (8,) - second polygon (4 points)

    Returns:
        IoU value as a scalar tensor
    """
    # Reshape if necessary
    if poly1.dim() == 1:
        poly1 = poly1.view(4, 2)
    if poly2.dim() == 1:
        poly2 = poly2.view(4, 2)

    # Compute areas
    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)

    if area1 < POLYGON_EPS or area2 < POLYGON_EPS:
        return torch.tensor(0.0, device=poly1.device, dtype=poly1.dtype)

    # Clip poly1 by poly2 to get intersection
    intersection_polygon = sutherland_hodgman_clip(poly1, poly2)

    if intersection_polygon.shape[0] < 3:
        return torch.tensor(0.0, device=poly1.device, dtype=poly1.dtype)

    intersection_area = polygon_area(intersection_polygon)
    union_area = area1 + area2 - intersection_area

    if union_area < POLYGON_EPS:
        return torch.tensor(0.0, device=poly1.device, dtype=poly1.dtype)

    iou = intersection_area / union_area
    return torch.clamp(iou, 0.0, 1.0)


def polygon_iou_batch(polys_a, polys_b):
    """
    Compute pairwise IoU between two sets of polygons.

    Args:
        polys_a: tensor of shape (N, 8) - N polygons, each with 4 points (x1,y1,x2,y2,x3,y3,x4,y4)
        polys_b: tensor of shape (M, 8) - M polygons, each with 4 points

    Returns:
        IoU matrix of shape (N, M)
    """
    n = polys_a.shape[0]
    m = polys_b.shape[0]
    iou_matrix = torch.zeros((n, m), device=polys_a.device, dtype=polys_a.dtype)

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = polygon_iou(polys_a[i], polys_b[j])

    return iou_matrix


def polygon_nms(polygons, scores, iou_threshold):
    """
    Non-Maximum Suppression for polygons.

    Args:
        polygons: tensor of shape (N, 8) - N polygons
        scores: tensor of shape (N,) - confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of kept polygons
    """
    if polygons.shape[0] == 0:
        return torch.tensor([], dtype=torch.long, device=polygons.device)

    # Sort by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)

    keep = []
    while sorted_indices.numel() > 0:
        # Pick the one with highest score
        current_idx = sorted_indices[0].item()
        keep.append(current_idx)

        if sorted_indices.numel() == 1:
            break

        # Compute IoU with remaining polygons
        current_poly = polygons[current_idx]
        remaining_indices = sorted_indices[1:]
        remaining_polys = polygons[remaining_indices]

        ious = torch.zeros(remaining_polys.shape[0], device=polygons.device, dtype=polygons.dtype)
        for i in range(remaining_polys.shape[0]):
            ious[i] = polygon_iou(current_poly, remaining_polys[i])

        # Keep only those with IoU below threshold
        mask = ious < iou_threshold
        sorted_indices = remaining_indices[mask]

    return torch.tensor(keep, dtype=torch.long, device=polygons.device)


def postprocess_polygon(
    prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False
):
    """
    Post-process polygon predictions.

    Args:
        prediction: tensor of shape (batch, n_anchors, 8 + 1 + num_classes)
                   where 8 is polygon coords, 1 is objectness
        num_classes: number of classes
        conf_thre: confidence threshold
        nms_thre: NMS IoU threshold
        class_agnostic: if True, perform class-agnostic NMS

    Returns:
        List of detections for each image, each detection is
        (x1,y1,x2,y2,x3,y3,x4,y4, obj_conf, class_conf, class_pred)
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

        # Detections: (polygons, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :9], class_conf, class_pred.float()), 1
        )
        detections = detections[conf_mask]

        if not detections.size(0):
            continue

        polygons = detections[:, :8]
        scores = detections[:, 8] * detections[:, 9]

        if class_agnostic:
            polygons = order_vertices_torch(polygons) 
            nms_out_index = polygon_nms(polygons, scores, nms_thre)
        else:
            # Batched NMS per class
            classes = detections[:, 10]
            unique_classes = torch.unique(classes)
            nms_out_indices = []
            for cls in unique_classes:
                cls_mask = classes == cls
                cls_polygons = polygons[cls_mask]
                cls_scores = scores[cls_mask]
                cls_indices = torch.where(cls_mask)[0]
                # Enforce ordering before NMS
                cls_polygons = order_vertices_torch(cls_polygons)
                cls_keep = polygon_nms(cls_polygons, cls_scores, nms_thre)
                nms_out_indices.append(cls_indices[cls_keep])
            if len(nms_out_indices) > 0:
                nms_out_index = torch.cat(nms_out_indices)
            else:
                nms_out_index = torch.tensor([], dtype=torch.long, device=detections.device)

        detections = detections[nms_out_index]

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output
