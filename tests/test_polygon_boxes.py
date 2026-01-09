#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import unittest
import numpy as np
import torch

from yolox.utils import polygon2xyxy, xyxy2polygon, polygon_iou, bboxes_iou


class TestPolygonBoxes(unittest.TestCase):

    def test_polygon2xyxy_torch(self):
        """Test converting polygon to xyxy format with torch tensors"""
        # Create a polygon: square from (10, 10) to (20, 20)
        # Points: top-left, top-right, bottom-right, bottom-left
        polygon = torch.tensor([[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]])
        expected_xyxy = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
        
        result = polygon2xyxy(polygon)
        self.assertTrue(torch.allclose(result, expected_xyxy))
    
    def test_polygon2xyxy_numpy(self):
        """Test converting polygon to xyxy format with numpy arrays"""
        # Create a polygon: square from (10, 10) to (20, 20)
        polygon = np.array([[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]])
        expected_xyxy = np.array([[10.0, 10.0, 20.0, 20.0]])
        
        result = polygon2xyxy(polygon)
        self.assertTrue(np.allclose(result, expected_xyxy))
    
    def test_xyxy2polygon_torch(self):
        """Test converting xyxy to polygon format with torch tensors"""
        xyxy = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
        # Expected: top-left, top-right, bottom-right, bottom-left
        expected_polygon = torch.tensor([[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]])
        
        result = xyxy2polygon(xyxy)
        self.assertTrue(torch.allclose(result, expected_polygon))
    
    def test_xyxy2polygon_numpy(self):
        """Test converting xyxy to polygon format with numpy arrays"""
        xyxy = np.array([[10.0, 10.0, 20.0, 20.0]])
        expected_polygon = np.array([[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]])
        
        result = xyxy2polygon(xyxy)
        self.assertTrue(np.allclose(result, expected_polygon))
    
    def test_polygon_iou_identical(self):
        """Test IoU of identical polygons should be 1.0"""
        polygon = torch.tensor([[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]])
        
        iou = polygon_iou(polygon, polygon)
        self.assertTrue(torch.allclose(iou, torch.tensor([[1.0]])))
    
    def test_polygon_iou_non_overlapping(self):
        """Test IoU of non-overlapping polygons should be 0.0"""
        polygon1 = torch.tensor([[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]])
        polygon2 = torch.tensor([[30.0, 30.0, 40.0, 30.0, 40.0, 40.0, 30.0, 40.0]])
        
        iou = polygon_iou(polygon1, polygon2)
        self.assertTrue(torch.allclose(iou, torch.tensor([[0.0]])))
    
    def test_polygon_iou_partial_overlap(self):
        """Test IoU of partially overlapping polygons"""
        # Square 1: (0, 0) to (10, 10)
        polygon1 = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        # Square 2: (5, 5) to (15, 15) - 50% overlap area = 25
        polygon2 = torch.tensor([[5.0, 5.0, 15.0, 5.0, 15.0, 15.0, 5.0, 15.0]])
        
        iou = polygon_iou(polygon1, polygon2)
        # IoU = intersection / union = 25 / (100 + 100 - 25) = 25 / 175 ≈ 0.143
        expected_iou = 25.0 / 175.0
        self.assertTrue(torch.allclose(iou, torch.tensor([[expected_iou]]), atol=1e-5))
    
    def test_bboxes_iou_with_polygons(self):
        """Test that bboxes_iou automatically handles polygon format"""
        polygon1 = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        polygon2 = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        
        iou = bboxes_iou(polygon1, polygon2)
        self.assertTrue(torch.allclose(iou, torch.tensor([[1.0]])))
    
    def test_polygon_conversion_roundtrip(self):
        """Test that converting xyxy->polygon->xyxy returns original"""
        original_xyxy = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        
        polygon = xyxy2polygon(original_xyxy)
        result_xyxy = polygon2xyxy(polygon)
        
        self.assertTrue(torch.allclose(original_xyxy, result_xyxy))
    
    def test_batch_polygon_conversion(self):
        """Test polygon conversion with batch of boxes"""
        xyxy_batch = torch.tensor([
            [10.0, 10.0, 20.0, 20.0],
            [30.0, 30.0, 40.0, 40.0],
            [50.0, 50.0, 60.0, 60.0]
        ])
        
        polygons = xyxy2polygon(xyxy_batch)
        result_xyxy = polygon2xyxy(polygons)
        
        self.assertTrue(torch.allclose(xyxy_batch, result_xyxy))


if __name__ == "__main__":
    unittest.main()
