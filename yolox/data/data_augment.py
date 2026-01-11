#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from yolox.utils import xyxy2cxcywh
from yolox.utils.boxes import polygon_area_np, polygon_centroid_np


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, use_polygon=False):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.use_polygon = use_polygon

    def __call__(self, image, targets, input_dim):
        reg_dim = 8 if self.use_polygon else 4
        ann = targets[:, :reg_dim].copy()
        labels = targets[:, reg_dim].copy()

        if len(ann) == 0:
            targets = np.zeros((self.max_labels, reg_dim + 1), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        ann_o = targets_o[:, :reg_dim]
        labels_o = targets_o[:, reg_dim]

        if self.use_polygon:
            # Polygons are already in pixel coordinates [x1, y1, ...]
            pass
        else:
            # bbox_o: [xyxy] to [c_x,c_y,w,h]
            ann_o = xyxy2cxcywh(ann_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)

        if self.use_polygon:
            image_t, ann = _mirror_polygon(image, ann, self.flip_prob)
        else:
            image_t, ann = _mirror(image, ann, self.flip_prob)

        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)

        if self.use_polygon:
            ann *= r_
            # Filter by area
            areas = polygon_area_np(ann)
            mask_b = areas > 1
        else:
            # boxes [xyxy] 2 [cx,cy,w,h]
            ann = xyxy2cxcywh(ann)
            ann *= r_
            mask_b = np.minimum(ann[:, 2], ann[:, 3]) > 1

        ann_t = ann[mask_b]
        labels_t = labels[mask_b]

        if len(ann_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            ann_o *= r_o
            ann_t = ann_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, ann_t))
        padded_labels = np.zeros((self.max_labels, reg_dim + 1))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False, use_polygon=False):
        self.swap = swap
        self.legacy = legacy
        self.use_polygon = use_polygon

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        
        reg_dim = 8 if self.use_polygon else 4
        return img, np.zeros((1, reg_dim + 1))


# ===================== Polygon Bounding Box Augmentations =====================


def apply_affine_to_polygons(targets, target_size, M, scale):
    """
    Apply affine transformation to polygon annotations.
    
    Args:
        targets: ndarray of shape (N, 9) where first 8 cols are polygon coords
                 (x1,y1, x2,y2, x3,y3, x4,y4) and 9th is class
        target_size: (width, height) of target image
        M: 2x3 affine transformation matrix
        scale: scale factor (unused but kept for API compatibility)
    
    Returns:
        Transformed targets with adjusted polygon coordinates.
    """
    num_gts = len(targets)
    if num_gts == 0:
        return targets
    
    twidth, theight = target_size
    
    # Transform all 4 corner points (8 coordinates)
    # Reshape to (N*4, 2) for matrix multiplication
    points = np.ones((num_gts * 4, 3))
    points[:, :2] = targets[:, :8].reshape(-1, 2)
    
    # Apply affine transform
    transformed = points @ M.T  # (N*4, 2)
    
    # Reshape back to (N, 8)
    new_coords = transformed[:, :2].reshape(num_gts, 8)
    
    # Clip to image bounds
    new_coords[:, 0::2] = new_coords[:, 0::2].clip(0, twidth)
    new_coords[:, 1::2] = new_coords[:, 1::2].clip(0, theight)
    
    targets[:, :8] = new_coords
    return targets


def random_affine_polygon(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    """
    Apply random affine transformation to image and polygon targets.
    
    Args:
        img: Input image
        targets: ndarray of shape (N, 9) - 8 polygon coords + 1 class
        target_size: (width, height) of output
        degrees: Rotation range
        translate: Translation range
        scales: Scale range
        shear: Shear range
    
    Returns:
        Transformed image and targets.
    """
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)
    
    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))
    
    # Transform polygon coordinates
    if len(targets) > 0:
        targets = apply_affine_to_polygons(targets, target_size, M, scale)
    
    return img, targets


def _mirror_polygon(image, polygons, prob=0.5):
    """
    Apply horizontal flip to image and polygon annotations.
    
    Args:
        image: Input image
        polygons: ndarray of shape (N, 8+) with polygon coordinates
        prob: Probability of applying mirror
    
    Returns:
        Flipped image and polygons with x-coordinates mirrored.
    """
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        # Mirror all x coordinates
        polygons[:, 0::2] = width - polygons[:, 0::2]
        # Reorder points to maintain consistent winding order
        # Original: (x1,y1), (x2,y2), (x3,y3), (x4,y4)
        # After flip: swap point pairs to maintain clockwise/counter-clockwise order
        # We swap columns: [0,1,2,3,4,5,6,7] -> [2,1,0,3,6,5,4,7]
        # This swaps p1<->p2 and p3<->p4 x-coordinates
        temp = polygons.copy()
        polygons[:, 0] = temp[:, 2]  # x2 -> x1
        polygons[:, 2] = temp[:, 0]  # x1 -> x2
        polygons[:, 4] = temp[:, 6]  # x4 -> x3
        polygons[:, 6] = temp[:, 4]  # x3 -> x4
    return image, polygons




class ValTransformPolygon:
    """
    Validation transform for polygon bounding boxes.
    """
    
    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    def __call__(self, img, res, input_size):
        """
        Apply validation preprocessing.
        
        Args:
            img: Input image
            res: Unused (for API compatibility)
            input_size: Target size
        
        Returns:
            Preprocessed image and empty polygon labels.
        """
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        # Return empty polygon labels (9 values: class + 8 coords)
        return img, np.zeros((1, 9))

