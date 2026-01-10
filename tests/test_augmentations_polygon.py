#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import unittest
import numpy as np
import cv2

from yolox.data.data_augment import (
    random_affine,
    apply_affine_to_bboxes,
    _mirror,
    TrainTransform,
    get_affine_matrix,
)
from yolox.utils import adjust_box_anns


class TestAugmentationsPolygon(unittest.TestCase):
    """Test that all data augmentations work with polygon format"""

    def setUp(self):
        """Set up test fixtures"""
        # Traditional format: [x1, y1, x2, y2, class]
        self.traditional_targets = np.array([
            [100, 100, 200, 200, 0],  # Square
            [300, 150, 400, 250, 1],  # Rectangle
        ], dtype=np.float32)
        
        # Polygon format: [x1, y1, x2, y2, x3, y3, x4, y4, class]
        self.polygon_targets = np.array([
            [100, 100, 200, 100, 200, 200, 100, 200, 0],  # Square
            [300, 150, 400, 150, 400, 250, 300, 250, 1],  # Rectangle
        ], dtype=np.float32)
        
        self.img = np.ones((480, 640, 3), dtype=np.uint8) * 114

    def test_apply_affine_to_bboxes_traditional(self):
        """Test affine transformation with traditional bbox format"""
        M, scale = get_affine_matrix((640, 640), degrees=10.0, translate=0.1, scales=0.1, shear=5.0)
        targets = self.traditional_targets.copy()
        result = apply_affine_to_bboxes(targets, (640, 640), M, scale)
        
        # Check that we still have 5 columns
        self.assertEqual(result.shape[1], 5)
        # Check that class labels are preserved
        np.testing.assert_array_equal(result[:, 4], self.traditional_targets[:, 4])
        # Check that coordinates changed (transformation applied)
        self.assertFalse(np.allclose(result[:, :4], self.traditional_targets[:, :4]))
    
    def test_apply_affine_to_bboxes_polygon(self):
        """Test affine transformation with polygon bbox format"""
        M, scale = get_affine_matrix((640, 640), degrees=10.0, translate=0.1, scales=0.1, shear=5.0)
        targets = self.polygon_targets.copy()
        result = apply_affine_to_bboxes(targets, (640, 640), M, scale)
        
        # Check that we still have 9 columns
        self.assertEqual(result.shape[1], 9)
        # Check that class labels are preserved
        np.testing.assert_array_equal(result[:, 8], self.polygon_targets[:, 8])
        # Check that coordinates changed (transformation applied)
        self.assertFalse(np.allclose(result[:, :8], self.polygon_targets[:, :8]))
    
    def test_mirror_traditional(self):
        """Test mirroring with traditional bbox format"""
        img = self.img.copy()
        boxes = self.traditional_targets[:, :4].copy()
        original_boxes = boxes.copy()
        
        # Force mirror to happen (prob=1.0)
        mirrored_img, mirrored_boxes = _mirror(img, boxes, prob=1.0)
        
        # Check image was flipped
        self.assertTrue(np.array_equal(mirrored_img, img[:, ::-1]))
        
        # Check that x coordinates were mirrored correctly
        # _mirror modifies in place, using the formula: x' = width - x
        width = img.shape[1]
        expected_x1 = width - original_boxes[:, 0]
        expected_x2 = width - original_boxes[:, 2]
        np.testing.assert_array_almost_equal(mirrored_boxes[:, 0], expected_x1)
        np.testing.assert_array_almost_equal(mirrored_boxes[:, 2], expected_x2)
        
        # Check that y coordinates are unchanged
        np.testing.assert_array_equal(mirrored_boxes[:, 1], original_boxes[:, 1])
        np.testing.assert_array_equal(mirrored_boxes[:, 3], original_boxes[:, 3])
    
    def test_mirror_polygon(self):
        """Test mirroring with polygon bbox format"""
        img = self.img.copy()
        boxes = self.polygon_targets[:, :8].copy()
        original_boxes = boxes.copy()
        
        # Force mirror to happen (prob=1.0)
        mirrored_img, mirrored_boxes = _mirror(img, boxes, prob=1.0)
        
        # Check image was flipped
        self.assertTrue(np.array_equal(mirrored_img, img[:, ::-1]))
        
        # Check that all x coordinates were mirrored correctly
        # _mirror modifies in place, using the formula: x' = width - x
        width = img.shape[1]
        for i in range(4):  # 4 points
            expected_x = width - original_boxes[:, i*2]
            np.testing.assert_array_almost_equal(mirrored_boxes[:, i*2], expected_x)
        
        # Check that all y coordinates are unchanged
        for i in range(4):  # 4 points
            np.testing.assert_array_equal(mirrored_boxes[:, i*2+1], original_boxes[:, i*2+1])
    
    def test_adjust_box_anns_traditional(self):
        """Test adjust_box_anns with traditional format"""
        boxes = self.traditional_targets[:, :4].copy()
        original_boxes = boxes.copy()
        scale_ratio = 0.5
        padw, padh = 10, 20
        w_max, h_max = 640, 480
        
        result = adjust_box_anns(boxes, scale_ratio, padw, padh, w_max, h_max)
        
        # Check that transformation was applied correctly
        # Note: adjust_box_anns modifies in place, so result is the same as boxes
        expected_x1 = np.clip(original_boxes[:, 0] * scale_ratio + padw, 0, w_max)
        expected_x2 = np.clip(original_boxes[:, 2] * scale_ratio + padw, 0, w_max)
        np.testing.assert_array_almost_equal(result[:, 0], expected_x1)
        np.testing.assert_array_almost_equal(result[:, 2], expected_x2)
    
    def test_adjust_box_anns_polygon(self):
        """Test adjust_box_anns with polygon format"""
        boxes = self.polygon_targets[:, :8].copy()
        original_boxes = boxes.copy()
        scale_ratio = 0.5
        padw, padh = 10, 20
        w_max, h_max = 640, 480
        
        result = adjust_box_anns(boxes, scale_ratio, padw, padh, w_max, h_max)
        
        # Check that all points were transformed correctly
        # Note: adjust_box_anns modifies in place, so result is the same as boxes
        for i in range(4):
            expected_x = np.clip(original_boxes[:, i*2] * scale_ratio + padw, 0, w_max)
            expected_y = np.clip(original_boxes[:, i*2+1] * scale_ratio + padh, 0, h_max)
            np.testing.assert_array_almost_equal(result[:, i*2], expected_x)
            np.testing.assert_array_almost_equal(result[:, i*2+1], expected_y)
    
    def test_random_affine_traditional(self):
        """Test random_affine with traditional format"""
        img = self.img.copy()
        targets = self.traditional_targets.copy()
        
        result_img, result_targets = random_affine(
            img, targets, target_size=(640, 640),
            degrees=10.0, translate=0.1, scales=0.1, shear=5.0
        )
        
        # Check output shape
        self.assertEqual(result_img.shape, (640, 640, 3))
        # Check that we still have 5 columns
        self.assertEqual(result_targets.shape[1], 5)
        # Check that class labels are preserved
        np.testing.assert_array_equal(result_targets[:, 4], targets[:, 4])
    
    def test_random_affine_polygon(self):
        """Test random_affine with polygon format"""
        img = self.img.copy()
        targets = self.polygon_targets.copy()
        
        result_img, result_targets = random_affine(
            img, targets, target_size=(640, 640),
            degrees=10.0, translate=0.1, scales=0.1, shear=5.0
        )
        
        # Check output shape
        self.assertEqual(result_img.shape, (640, 640, 3))
        # Check that we still have 9 columns
        self.assertEqual(result_targets.shape[1], 9)
        # Check that class labels are preserved
        np.testing.assert_array_equal(result_targets[:, 8], targets[:, 8])
    
    def test_train_transform_traditional(self):
        """Test TrainTransform with traditional format"""
        transform = TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=0.5)
        img = self.img.copy()
        targets = self.traditional_targets.copy()
        
        result_img, result_labels = transform(img, targets, input_dim=(640, 640))
        
        # Check output shape
        self.assertEqual(result_img.shape, (3, 640, 640))
        # Check padded labels shape (max_labels, 5)
        self.assertEqual(result_labels.shape, (50, 5))
        # Check that at least some labels exist
        non_zero_rows = np.any(result_labels != 0, axis=1).sum()
        self.assertGreater(non_zero_rows, 0)
    
    def test_train_transform_polygon(self):
        """Test TrainTransform with polygon format"""
        transform = TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=0.5)
        img = self.img.copy()
        targets = self.polygon_targets.copy()
        
        result_img, result_labels = transform(img, targets, input_dim=(640, 640))
        
        # Check output shape
        self.assertEqual(result_img.shape, (3, 640, 640))
        # Check padded labels shape (max_labels, 9)
        self.assertEqual(result_labels.shape, (50, 9))
        # Check that at least some labels exist
        non_zero_rows = np.any(result_labels != 0, axis=1).sum()
        self.assertGreater(non_zero_rows, 0)
    
    def test_coordinate_clipping_polygon(self):
        """Test that coordinate clipping works with polygon format using ::2 indexing"""
        boxes = np.array([
            [-10, 50, 700, 100, 650, 500, -5, 490],  # Out of bounds
            [100, 100, 200, 100, 200, 200, 100, 200],  # In bounds
        ], dtype=np.float32)
        
        w_max, h_max = 640, 480
        
        # Clip using ::2 indexing (like in the augmentations)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w_max)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h_max)
        
        # Check that all coordinates are within bounds
        self.assertTrue(np.all(boxes[:, 0::2] >= 0))
        self.assertTrue(np.all(boxes[:, 0::2] <= w_max))
        self.assertTrue(np.all(boxes[:, 1::2] >= 0))
        self.assertTrue(np.all(boxes[:, 1::2] <= h_max))
        
        # Check specific clipped values
        self.assertEqual(boxes[0, 0], 0)  # -10 clipped to 0
        self.assertEqual(boxes[0, 2], 640)  # 700 clipped to 640
        self.assertEqual(boxes[0, 4], 640)  # 650 clipped to 640
        self.assertEqual(boxes[0, 6], 0)  # -5 clipped to 0
        self.assertEqual(boxes[0, 5], 480)  # 500 clipped to 480
        self.assertEqual(boxes[0, 7], 480)  # 490 clipped to 480


if __name__ == "__main__":
    unittest.main()
