#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import unittest
import torch
import numpy as np

from yolox.utils import polygon_iou, polygon_area, polygon_to_bbox
from yolox.models.losses import PolygonIOULoss


class TestPolygonFunctions(unittest.TestCase):

    def test_polygon_area_square(self):
        """Test polygon area calculation for a square"""
        # Square with side length 10 (area = 100)
        square = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        area = polygon_area(square)
        self.assertAlmostEqual(area.item(), 100.0, places=1)

    def test_polygon_area_rectangle(self):
        """Test polygon area calculation for a rectangle"""
        # Rectangle 5x10 (area = 50)
        rectangle = torch.tensor([[0.0, 0.0, 5.0, 0.0, 5.0, 10.0, 0.0, 10.0]])
        area = polygon_area(rectangle)
        self.assertAlmostEqual(area.item(), 50.0, places=1)

    def test_polygon_iou_identical(self):
        """Test IoU of identical polygons should be 1.0"""
        poly1 = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        poly2 = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        iou = polygon_iou(poly1, poly2)
        self.assertAlmostEqual(iou[0, 0].item(), 1.0, places=3)

    def test_polygon_iou_no_overlap(self):
        """Test IoU of non-overlapping polygons should be 0.0"""
        poly1 = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        poly2 = torch.tensor([[20.0, 20.0, 30.0, 20.0, 30.0, 30.0, 20.0, 30.0]])
        iou = polygon_iou(poly1, poly2)
        self.assertAlmostEqual(iou[0, 0].item(), 0.0, places=3)

    def test_polygon_iou_partial_overlap(self):
        """Test IoU of partially overlapping polygons"""
        # Two squares, half overlapping
        poly1 = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        poly2 = torch.tensor([[5.0, 0.0, 15.0, 0.0, 15.0, 10.0, 5.0, 10.0]])
        iou = polygon_iou(poly1, poly2)
        # Intersection area = 50, Union area = 150, IoU = 50/150 = 0.333
        self.assertAlmostEqual(iou[0, 0].item(), 0.333, places=2)

    def test_polygon_iou_batch(self):
        """Test IoU calculation for batches of polygons"""
        polys_a = torch.tensor([
            [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
            [0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 5.0]
        ])
        polys_b = torch.tensor([
            [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
            [20.0, 20.0, 30.0, 20.0, 30.0, 30.0, 20.0, 30.0]
        ])
        iou_matrix = polygon_iou(polys_a, polys_b)
        
        # Check shape
        self.assertEqual(iou_matrix.shape, (2, 2))
        
        # Check diagonal elements
        self.assertAlmostEqual(iou_matrix[0, 0].item(), 1.0, places=3)
        self.assertAlmostEqual(iou_matrix[1, 1].item(), 0.0, places=3)

    def test_polygon_iou_loss(self):
        """Test PolygonIOULoss computation"""
        loss_fn = PolygonIOULoss(reduction="none")
        
        # Identical polygons should have loss close to 0
        pred = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        target = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        loss = loss_fn(pred, target)
        self.assertLess(loss.item(), 0.1)  # Loss should be close to 0

    def test_polygon_iou_loss_batch(self):
        """Test PolygonIOULoss with batches"""
        loss_fn = PolygonIOULoss(reduction="mean")
        
        pred = torch.tensor([
            [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
            [5.0, 5.0, 15.0, 5.0, 15.0, 15.0, 5.0, 15.0]
        ])
        target = torch.tensor([
            [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
            [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
        ])
        
        loss = loss_fn(pred, target)
        
        # Loss should be a scalar
        self.assertEqual(loss.shape, torch.Size([]))
        # Loss should be positive
        self.assertGreater(loss.item(), 0.0)

    def test_polygon_area_numpy(self):
        """Test polygon area calculation with numpy arrays"""
        square = np.array([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        area = polygon_area(square)
        self.assertAlmostEqual(area[0], 100.0, places=1)

    def test_polygon_iou_numpy(self):
        """Test polygon IoU with numpy arrays"""
        poly1 = np.array([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        poly2 = np.array([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        iou = polygon_iou(poly1, poly2, use_torch=False)
        self.assertAlmostEqual(iou[0, 0], 1.0, places=3)

    def test_polygon_to_bbox_torch(self):
        """Test polygon to bounding box conversion with torch tensors"""
        # Square polygon
        polygon = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        bbox = polygon_to_bbox(polygon)
        expected = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        self.assertTrue(torch.allclose(bbox, expected))
        
        # Rotated rectangle
        polygon = torch.tensor([[5.0, 0.0, 10.0, 5.0, 5.0, 10.0, 0.0, 5.0]])
        bbox = polygon_to_bbox(polygon)
        expected = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        self.assertTrue(torch.allclose(bbox, expected))

    def test_polygon_to_bbox_numpy(self):
        """Test polygon to bounding box conversion with numpy arrays"""
        polygon = np.array([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
        bbox = polygon_to_bbox(polygon)
        expected = np.array([[0.0, 0.0, 10.0, 10.0]])
        np.testing.assert_array_almost_equal(bbox, expected)

    def test_polygon_to_bbox_batch(self):
        """Test polygon to bounding box conversion with batches"""
        polygons = torch.tensor([
            [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
            [5.0, 5.0, 15.0, 5.0, 15.0, 15.0, 5.0, 15.0]
        ])
        bboxes = polygon_to_bbox(polygons)
        expected = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 15.0]
        ])
        self.assertTrue(torch.allclose(bboxes, expected))


if __name__ == "__main__":
    unittest.main()
