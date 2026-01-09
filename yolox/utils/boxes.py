#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision
from shapely.geometry import Polygon
from shapely.validation import make_valid

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy",
    "polygon_iou",
    "polygon_area",
    "polygon_to_bbox",
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
        raise ValueError("Bounding boxes must have 4 values (x1, y1, x2, y2) or (cx, cy, w, h)")

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


def polygon_area(vertices):
    """
    Calculate the area of a polygon given its vertices.
    
    Args:
        vertices: (N, 8) tensor or array representing N polygons with 4 vertices each (x1,y1,x2,y2,x3,y3,x4,y4)
    
    Returns:
        areas: (N,) tensor or array of polygon areas
    """
    if isinstance(vertices, torch.Tensor):
        # Reshape to (N, 4, 2) for 4 vertices with (x, y) coordinates
        vertices = vertices.view(-1, 4, 2)
        # Use Shoelace formula for area calculation
        x = vertices[:, :, 0]
        y = vertices[:, :, 1]
        # Roll to get next vertex coordinates
        x_next = torch.roll(x, -1, dims=1)
        y_next = torch.roll(y, -1, dims=1)
        # Calculate area using cross product
        area = 0.5 * torch.abs((x * y_next - x_next * y).sum(dim=1))
        return area
    else:
        # NumPy version
        vertices = vertices.reshape(-1, 4, 2)
        x = vertices[:, :, 0]
        y = vertices[:, :, 1]
        x_next = np.roll(x, -1, axis=1)
        y_next = np.roll(y, -1, axis=1)
        area = 0.5 * np.abs((x * y_next - x_next * y).sum(axis=1))
        return area


def polygon_iou(polygons_a, polygons_b, use_torch=True):
    """
    Calculate IoU between two sets of polygons.
    
    Args:
        polygons_a: (N, 8) tensor or array representing N polygons with 4 vertices each
        polygons_b: (M, 8) tensor or array representing M polygons with 4 vertices each
        use_torch: whether to return torch tensor (True) or numpy array (False)
    
    Returns:
        iou_matrix: (N, M) tensor or array of IoU values
    """
    if polygons_a.shape[1] != 8 or polygons_b.shape[1] != 8:
        raise ValueError("Polygons must have 8 values (4 vertices with x,y coordinates)")
    
    # Convert to numpy for Shapely processing
    if isinstance(polygons_a, torch.Tensor):
        polygons_a_np = polygons_a.detach().cpu().numpy()
        polygons_b_np = polygons_b.detach().cpu().numpy()
        device = polygons_a.device
        dtype = polygons_a.dtype
    else:
        polygons_a_np = polygons_a
        polygons_b_np = polygons_b
        device = None
        dtype = None
    
    n_a = polygons_a_np.shape[0]
    n_b = polygons_b_np.shape[0]
    iou_matrix = np.zeros((n_a, n_b), dtype=np.float32)
    
    # Pre-create and validate polygons for set A
    polys_a = []
    areas_a = []
    for i in range(n_a):
        coords_a = polygons_a_np[i].reshape(4, 2)
        try:
            poly_a = Polygon(coords_a)
            if not poly_a.is_valid:
                poly_a = make_valid(poly_a)
            polys_a.append(poly_a)
            areas_a.append(poly_a.area)
        except Exception:
            # Degenerate polygon
            polys_a.append(None)
            areas_a.append(0.0)
    
    # Pre-create and validate polygons for set B
    polys_b = []
    areas_b = []
    for j in range(n_b):
        coords_b = polygons_b_np[j].reshape(4, 2)
        try:
            poly_b = Polygon(coords_b)
            if not poly_b.is_valid:
                poly_b = make_valid(poly_b)
            polys_b.append(poly_b)
            areas_b.append(poly_b.area)
        except Exception:
            # Degenerate polygon
            polys_b.append(None)
            areas_b.append(0.0)
    
    # Compute IoU matrix
    for i in range(n_a):
        if polys_a[i] is None:
            continue
        area_a = areas_a[i]
        
        for j in range(n_b):
            if polys_b[j] is None:
                continue
            area_b = areas_b[j]
            
            try:
                # Calculate intersection
                if polys_a[i].intersects(polys_b[j]):
                    intersection = polys_a[i].intersection(polys_b[j])
                    area_i = intersection.area
                else:
                    area_i = 0.0
                
                # Calculate IoU
                area_u = area_a + area_b - area_i
                if area_u > 0:
                    iou_matrix[i, j] = area_i / area_u
                else:
                    iou_matrix[i, j] = 0.0
                    
            except Exception:
                # Handle any unexpected errors
                iou_matrix[i, j] = 0.0
    
    if use_torch and device is not None:
        return torch.from_numpy(iou_matrix).to(device=device, dtype=dtype)
    else:
        return iou_matrix


def polygon_to_bbox(polygons):
    """
    Convert polygon vertices to axis-aligned bounding boxes.
    
    Args:
        polygons: (N, 8) tensor or array representing N polygons with 4 vertices each
                  Format: (x1, y1, x2, y2, x3, y3, x4, y4)
    
    Returns:
        bboxes: (N, 4) tensor or array in xyxy format (x_min, y_min, x_max, y_max)
    """
    if isinstance(polygons, torch.Tensor):
        # Reshape to (N, 4, 2) for 4 vertices with (x, y) coordinates
        vertices = polygons.view(-1, 4, 2)
        x_coords = vertices[:, :, 0]
        y_coords = vertices[:, :, 1]
        
        x_min = x_coords.min(dim=1)[0]
        y_min = y_coords.min(dim=1)[0]
        x_max = x_coords.max(dim=1)[0]
        y_max = y_coords.max(dim=1)[0]
        
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)
    else:
        # NumPy version
        vertices = polygons.reshape(-1, 4, 2)
        x_coords = vertices[:, :, 0]
        y_coords = vertices[:, :, 1]
        
        x_min = x_coords.min(axis=1)
        y_min = y_coords.min(axis=1)
        x_max = x_coords.max(axis=1)
        y_max = y_coords.max(axis=1)
        
        return np.stack([x_min, y_min, x_max, y_max], axis=1)
