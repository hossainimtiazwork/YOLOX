#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import unittest

import torch

from yolox.utils import (
    polygon_area,
    sutherland_hodgman_clip,
    polygon_iou,
    polygon_iou_batch,
    polygon_nms,
)
from yolox.models import PolygonIOULoss


class TestPolygonArea(unittest.TestCase):

    def test_square_area(self):
        """Test area of a 2x2 square."""
        square = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ])
        area = polygon_area(square)
        self.assertAlmostEqual(area.item(), 4.0, places=5)

    def test_triangle_area(self):
        """Test area of a triangle."""
        triangle = torch.tensor([
            [0.0, 0.0],
            [4.0, 0.0],
            [2.0, 3.0],
        ])
        area = polygon_area(triangle)
        self.assertAlmostEqual(area.item(), 6.0, places=5)

    def test_rectangle_area(self):
        """Test area of a 3x4 rectangle."""
        rectangle = torch.tensor([
            [1.0, 1.0],
            [4.0, 1.0],
            [4.0, 5.0],
            [1.0, 5.0],
        ])
        area = polygon_area(rectangle)
        self.assertAlmostEqual(area.item(), 12.0, places=5)


class TestSutherlandHodgman(unittest.TestCase):

    def test_clip_contained_square(self):
        """Clipping a square by a larger square should return the original."""
        small_square = torch.tensor([
            [1.0, 1.0],
            [3.0, 1.0],
            [3.0, 3.0],
            [1.0, 3.0],
        ])
        large_square = torch.tensor([
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
        ])
        clipped = sutherland_hodgman_clip(small_square, large_square)
        clipped_area = polygon_area(clipped)
        self.assertAlmostEqual(clipped_area.item(), 4.0, places=5)

    def test_partial_overlap(self):
        """Test partial overlap between two squares."""
        square1 = torch.tensor([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ])
        square2 = torch.tensor([
            [1.0, 1.0],
            [3.0, 1.0],
            [3.0, 3.0],
            [1.0, 3.0],
        ])
        clipped = sutherland_hodgman_clip(square1, square2)
        clipped_area = polygon_area(clipped)
        self.assertAlmostEqual(clipped_area.item(), 1.0, places=5)


class TestPolygonIoU(unittest.TestCase):

    def test_identical_squares(self):
        """Identical squares should have IoU of 1.0."""
        square = torch.tensor([0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0])
        iou = polygon_iou(square, square)
        self.assertAlmostEqual(iou.item(), 1.0, places=5)

    def test_non_overlapping_squares(self):
        """Non-overlapping squares should have IoU of 0.0."""
        square1 = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        square2 = torch.tensor([5.0, 5.0, 6.0, 5.0, 6.0, 6.0, 5.0, 6.0])
        iou = polygon_iou(square1, square2)
        self.assertAlmostEqual(iou.item(), 0.0, places=5)

    def test_partial_overlap(self):
        """Test IoU for partially overlapping squares."""
        square1 = torch.tensor([0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0])
        square2 = torch.tensor([1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0])
        iou = polygon_iou(square1, square2)
        # Intersection: 2, Union: 6, IoU = 2/6 = 0.333
        self.assertAlmostEqual(iou.item(), 2.0 / 6.0, places=2)


class TestPolygonIoUBatch(unittest.TestCase):

    def test_batch_shape(self):
        """Test that batch IoU returns correct shape."""
        polys_a = torch.tensor([
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
            [5.0, 5.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0],
        ])
        polys_b = torch.tensor([
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
            [1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0],
        ])
        iou_matrix = polygon_iou_batch(polys_a, polys_b)
        self.assertEqual(iou_matrix.shape, (2, 2))

    def test_batch_values(self):
        """Test batch IoU values."""
        polys_a = torch.tensor([
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
            [5.0, 5.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0],
        ])
        polys_b = torch.tensor([
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
        ])
        iou_matrix = polygon_iou_batch(polys_a, polys_b)
        # First polygon identical, second non-overlapping
        self.assertAlmostEqual(iou_matrix[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(iou_matrix[1, 0].item(), 0.0, places=5)


class TestPolygonNMS(unittest.TestCase):

    def test_high_threshold(self):
        """With high threshold, all polygons should be kept."""
        polygons = torch.tensor([
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
            [0.5, 0.0, 2.5, 0.0, 2.5, 2.0, 0.5, 2.0],
            [5.0, 5.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0],
        ])
        scores = torch.tensor([0.9, 0.7, 0.8])
        keep = polygon_nms(polygons, scores, iou_threshold=0.9)
        self.assertEqual(len(keep), 3)

    def test_low_threshold(self):
        """With low threshold, overlapping polygons should be suppressed."""
        polygons = torch.tensor([
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
            [0.5, 0.0, 2.5, 0.0, 2.5, 2.0, 0.5, 2.0],
            [5.0, 5.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0],
        ])
        scores = torch.tensor([0.9, 0.7, 0.8])
        keep = polygon_nms(polygons, scores, iou_threshold=0.3)
        self.assertEqual(len(keep), 2)
        self.assertIn(0, keep.tolist())
        self.assertIn(2, keep.tolist())

    def test_empty_input(self):
        """Empty input should return empty tensor."""
        polygons = torch.zeros((0, 8))
        scores = torch.zeros(0)
        keep = polygon_nms(polygons, scores, iou_threshold=0.5)
        self.assertEqual(len(keep), 0)


class TestPolygonIOULoss(unittest.TestCase):

    def test_identical_polygons(self):
        """Identical polygons should have loss of 0."""
        loss_fn = PolygonIOULoss(reduction="mean")
        pred = torch.tensor([[0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]])
        target = torch.tensor([[0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]])
        loss = loss_fn(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_non_overlapping_polygons(self):
        """Non-overlapping polygons should have loss of 1."""
        loss_fn = PolygonIOULoss(reduction="mean")
        pred = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]])
        target = torch.tensor([[5.0, 5.0, 6.0, 5.0, 6.0, 6.0, 5.0, 6.0]])
        loss = loss_fn(pred, target)
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_reduction_sum(self):
        """Test sum reduction."""
        loss_fn = PolygonIOULoss(reduction="sum")
        pred = torch.tensor([
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
        ])
        target = torch.tensor([
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
        ])
        loss = loss_fn(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
