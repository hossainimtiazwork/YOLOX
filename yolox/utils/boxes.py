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
    # Check if polygon format (8 coords + obj + classes) or bbox format (4 coords + obj + classes)
    if prediction.shape[2] >= 9 + num_classes:
        # Polygon format - convert to bbox for NMS
        polygon_coords = prediction[:, :, :8].clone()
        
        # Convert polygon to bounding box for NMS
        x_coords = polygon_coords[:, :, [0, 2, 4, 6]]
        y_coords = polygon_coords[:, :, [1, 3, 5, 7]]
        min_x = x_coords.min(dim=2)[0]
        max_x = x_coords.max(dim=2)[0]
        min_y = y_coords.min(dim=2)[0]
        max_y = y_coords.max(dim=2)[0]
        
        box_corner = torch.stack([min_x, min_y, max_x, max_y], dim=2)
        
        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 9: 9 + num_classes], 1, keepdim=True)
            
            conf_mask = (image_pred[:, 8] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (8 polygon coords, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :9], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            bbox_for_nms = box_corner[i][conf_mask]
            
            if not detections.size(0):
                continue
            
            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    bbox_for_nms,
                    detections[:, 8] * detections[:, 9],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    bbox_for_nms,
                    detections[:, 8] * detections[:, 9],
                    detections[:, 10],
                    nms_thre,
                )
            
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
    else:
        # Original bbox format
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
    # Check if polygon format (8 coordinates) or bbox format (4 coordinates)
    if bboxes_a.shape[1] == 8 and bboxes_b.shape[1] == 8:
        # Polygon format - convert to bounding boxes for IoU calculation
        def polygon_to_bbox(coords):
            x_coords = coords[:, [0, 2, 4, 6]]
            y_coords = coords[:, [1, 3, 5, 7]]
            min_x = x_coords.min(dim=1)[0]
            max_x = x_coords.max(dim=1)[0]
            min_y = y_coords.min(dim=1)[0]
            max_y = y_coords.max(dim=1)[0]
            return torch.stack([min_x, min_y, max_x, max_y], dim=1)
        
        bbox_a = polygon_to_bbox(bboxes_a)
        bbox_b = polygon_to_bbox(bboxes_b)
        
        tl = torch.max(bbox_a[:, None, :2], bbox_b[:, :2])
        br = torch.min(bbox_a[:, None, 2:], bbox_b[:, 2:])
        area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], 1)
        area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], 1)
    elif bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError("Bounding boxes must have 4 or 8 coordinates")
    elif xyxy:
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
