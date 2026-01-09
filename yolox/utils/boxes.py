#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy",
    "polygon2xyxy",
    "xyxy2polygon",
    "polygon_iou",
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
    # Check if we have polygon format (8 values) or traditional format (4 values)
    is_polygon = prediction.shape[2] >= 9  # 8 for bbox + at least 1 for obj
    
    if is_polygon:
        # Convert polygon to xyxy for NMS
        box_corner = prediction.new(prediction.shape[0], prediction.shape[1], 4)
        polygons = prediction[:, :, :8]
        x_coords = polygons[:, :, [0, 2, 4, 6]]
        y_coords = polygons[:, :, [1, 3, 5, 7]]
        box_corner[:, :, 0] = torch.min(x_coords, dim=2)[0]
        box_corner[:, :, 1] = torch.min(y_coords, dim=2)[0]
        box_corner[:, :, 2] = torch.max(x_coords, dim=2)[0]
        box_corner[:, :, 3] = torch.max(y_coords, dim=2)[0]
        # Create new prediction with xyxy boxes
        prediction_nms = torch.cat([box_corner, prediction[:, :, 8:]], dim=2)
        obj_start_idx = 4
        cls_start_idx = 5
        bbox_size = 4
    else:
        # Traditional format
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction_nms = box_corner
        obj_start_idx = 4
        cls_start_idx = 5
        bbox_size = 4

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction_nms):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, cls_start_idx: cls_start_idx + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, obj_start_idx] * class_conf.squeeze() >= conf_thre).squeeze()
        
        if is_polygon:
            # For polygon, return original polygon coordinates with scores
            orig_pred = prediction[i]
            detections = torch.cat((orig_pred[:, :8], image_pred[:, 4:5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            # For NMS, use xyxy boxes
            nms_boxes = image_pred[conf_mask, :4]
        else:
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            nms_boxes = detections[:, :4]
            
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                nms_boxes,
                detections[:, -3] * detections[:, -2],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                nms_boxes,
                detections[:, -3] * detections[:, -2],
                detections[:, -1],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    # Handle polygon format (8 values)
    if bboxes_a.shape[1] == 8 and bboxes_b.shape[1] == 8:
        return polygon_iou(bboxes_a, bboxes_b)
    
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


def polygon2xyxy(polygons):
    """
    Convert polygon format (x1, y1, x2, y2, x3, y3, x4, y4) to xyxy format.
    Args:
        polygons: tensor/array of shape [..., 8]
    Returns:
        bboxes: tensor/array of shape [..., 4] in xyxy format
    """
    if isinstance(polygons, torch.Tensor):
        x_coords = polygons[..., [0, 2, 4, 6]]
        y_coords = polygons[..., [1, 3, 5, 7]]
        x1 = torch.min(x_coords, dim=-1)[0]
        y1 = torch.min(y_coords, dim=-1)[0]
        x2 = torch.max(x_coords, dim=-1)[0]
        y2 = torch.max(y_coords, dim=-1)[0]
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        x_coords = polygons[..., [0, 2, 4, 6]]
        y_coords = polygons[..., [1, 3, 5, 7]]
        x1 = np.min(x_coords, axis=-1)
        y1 = np.min(y_coords, axis=-1)
        x2 = np.max(x_coords, axis=-1)
        y2 = np.max(y_coords, axis=-1)
        return np.stack([x1, y1, x2, y2], axis=-1)


def xyxy2polygon(bboxes):
    """
    Convert xyxy format to polygon format (4 corners).
    Args:
        bboxes: tensor/array of shape [..., 4] in xyxy format
    Returns:
        polygons: tensor/array of shape [..., 8] (x1, y1, x2, y2, x3, y3, x4, y4)
                 representing: top-left, top-right, bottom-right, bottom-left
    """
    if isinstance(bboxes, torch.Tensor):
        x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        return torch.stack([x1, y1, x2, y1, x2, y2, x1, y2], dim=-1)
    else:
        x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        return np.stack([x1, y1, x2, y1, x2, y2, x1, y2], axis=-1)


def polygon_iou(polygons_a, polygons_b):
    """
    Calculate IoU between polygons by converting to bounding boxes.
    
    Note: This is an approximation that uses axis-aligned bounding boxes.
    For simple convex quadrilaterals, this provides a reasonable approximation.
    For more accurate IoU calculation of arbitrary polygons, consider using
    specialized geometric libraries like Shapely.
    
    Args:
        polygons_a: tensor of shape [N, 8]
        polygons_b: tensor of shape [M, 8]
    Returns:
        iou: tensor of shape [N, M]
    """
    # Convert polygons to xyxy bounding boxes
    bboxes_a = polygon2xyxy(polygons_a)
    bboxes_b = polygon2xyxy(polygons_b)
    
    # Use existing bboxes_iou function
    return bboxes_iou(bboxes_a, bboxes_b, xyxy=True)

