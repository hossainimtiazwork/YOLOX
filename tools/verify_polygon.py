#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

"""
Verification script for polygon IoU and NMS logic.

This script tests the Sutherland-Hodgman polygon clipping algorithm and
polygon IoU computation to ensure correct behavior.

Usage:
    python tools/verify_polygon.py
"""

import torch
import numpy as np


def test_polygon_area():
    """Test polygon area computation."""
    from yolox.utils import polygon_area

    # Test with a simple square (side 2, area should be 4)
    square = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 2.0],
    ])
    area = polygon_area(square)
    expected = 4.0
    assert abs(area.item() - expected) < 1e-5, f"Square area: expected {expected}, got {area.item()}"
    print(f"✓ Square area test passed: {area.item():.4f}")

    # Test with a triangle (base 4, height 3, area should be 6)
    triangle = torch.tensor([
        [0.0, 0.0],
        [4.0, 0.0],
        [2.0, 3.0],
    ])
    area = polygon_area(triangle)
    expected = 6.0
    assert abs(area.item() - expected) < 1e-5, f"Triangle area: expected {expected}, got {area.item()}"
    print(f"✓ Triangle area test passed: {area.item():.4f}")

    # Test with a rectangle (3x4, area should be 12)
    rectangle = torch.tensor([
        [1.0, 1.0],
        [4.0, 1.0],
        [4.0, 5.0],
        [1.0, 5.0],
    ])
    area = polygon_area(rectangle)
    expected = 12.0
    assert abs(area.item() - expected) < 1e-5, f"Rectangle area: expected {expected}, got {area.item()}"
    print(f"✓ Rectangle area test passed: {area.item():.4f}")


def test_sutherland_hodgman():
    """Test Sutherland-Hodgman polygon clipping."""
    from yolox.utils import sutherland_hodgman_clip, polygon_area

    # Test 1: Clip a square by a larger square (should return original)
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
    expected_area = 4.0  # 2x2 square
    assert abs(clipped_area.item() - expected_area) < 1e-5, \
        f"Clipping contained square: expected area {expected_area}, got {clipped_area.item()}"
    print(f"✓ Clipping contained square test passed: area = {clipped_area.item():.4f}")

    # Test 2: Partial overlap
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
    expected_area = 1.0  # 1x1 overlap
    assert abs(clipped_area.item() - expected_area) < 1e-5, \
        f"Partial overlap: expected area {expected_area}, got {clipped_area.item()}"
    print(f"✓ Partial overlap test passed: area = {clipped_area.item():.4f}")


def test_polygon_iou():
    """Test polygon IoU computation."""
    from yolox.utils import polygon_iou

    # Test 1: Identical squares (IoU should be 1.0)
    square = torch.tensor([0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0])
    iou = polygon_iou(square, square)
    expected = 1.0
    assert abs(iou.item() - expected) < 1e-5, f"Identical squares IoU: expected {expected}, got {iou.item()}"
    print(f"✓ Identical squares IoU test passed: {iou.item():.4f}")

    # Test 2: Non-overlapping squares (IoU should be 0.0)
    square1 = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    square2 = torch.tensor([5.0, 5.0, 6.0, 5.0, 6.0, 6.0, 5.0, 6.0])
    iou = polygon_iou(square1, square2)
    expected = 0.0
    assert abs(iou.item() - expected) < 1e-5, f"Non-overlapping squares IoU: expected {expected}, got {iou.item()}"
    print(f"✓ Non-overlapping squares IoU test passed: {iou.item():.4f}")

    # Test 3: 50% overlapping squares
    # Square 1: (0,0) to (2,2), area = 4
    # Square 2: (1,0) to (3,2), area = 4
    # Intersection: (1,0) to (2,2), area = 2
    # Union: 4 + 4 - 2 = 6
    # IoU = 2/6 = 0.333...
    square1 = torch.tensor([0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0])
    square2 = torch.tensor([1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0])
    iou = polygon_iou(square1, square2)
    expected = 2.0 / 6.0
    assert abs(iou.item() - expected) < 0.01, f"50% overlapping squares IoU: expected {expected:.4f}, got {iou.item():.4f}"
    print(f"✓ 50% overlapping squares IoU test passed: {iou.item():.4f} (expected ~{expected:.4f})")


def test_polygon_iou_batch():
    """Test batch polygon IoU computation."""
    from yolox.utils import polygon_iou_batch

    # Create some test polygons
    polys_a = torch.tensor([
        [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],  # 2x2 square at origin
        [5.0, 5.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0],  # 2x2 square at (5,5)
    ])
    polys_b = torch.tensor([
        [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],  # Same as first in polys_a
        [1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0],  # Overlapping with first
    ])

    iou_matrix = polygon_iou_batch(polys_a, polys_b)
    assert iou_matrix.shape == (2, 2), f"IoU matrix shape: expected (2, 2), got {iou_matrix.shape}"

    # polys_a[0] and polys_b[0] are identical -> IoU = 1.0
    assert abs(iou_matrix[0, 0].item() - 1.0) < 1e-5, f"IoU[0,0]: expected 1.0, got {iou_matrix[0, 0].item()}"

    # polys_a[1] and polys_b[0] are non-overlapping -> IoU = 0.0
    assert abs(iou_matrix[1, 0].item() - 0.0) < 1e-5, f"IoU[1,0]: expected 0.0, got {iou_matrix[1, 0].item()}"

    print(f"✓ Batch IoU test passed")
    print(f"  IoU matrix:\n{iou_matrix}")


def test_polygon_nms():
    """Test polygon NMS."""
    from yolox.utils import polygon_nms

    # Create test polygons with varying scores
    # Three squares, first two overlap significantly, third is separate
    polygons = torch.tensor([
        [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0],  # Square 1
        [0.5, 0.0, 2.5, 0.0, 2.5, 2.0, 0.5, 2.0],  # Square 2 (overlaps with 1)
        [5.0, 5.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0],  # Square 3 (separate)
    ])
    scores = torch.tensor([0.9, 0.7, 0.8])

    # With high IoU threshold, all should be kept
    keep = polygon_nms(polygons, scores, iou_threshold=0.9)
    assert len(keep) == 3, f"High threshold NMS: expected 3 kept, got {len(keep)}"
    print(f"✓ High threshold NMS test passed: kept {len(keep)} polygons")

    # With lower threshold, overlapping polygons should be suppressed
    keep = polygon_nms(polygons, scores, iou_threshold=0.3)
    # Should keep square 1 (highest score) and square 3 (no overlap)
    assert len(keep) == 2, f"Low threshold NMS: expected 2 kept, got {len(keep)}"
    assert 0 in keep.tolist(), "Highest scoring polygon should be kept"
    assert 2 in keep.tolist(), "Non-overlapping polygon should be kept"
    print(f"✓ Low threshold NMS test passed: kept {len(keep)} polygons (indices: {keep.tolist()})")


def test_polygon_iou_loss():
    """Test PolygonIOULoss."""
    from yolox.models import PolygonIOULoss

    loss_fn = PolygonIOULoss(reduction="mean")

    # Test with identical polygons (loss should be 0 since IoU=1)
    pred = torch.tensor([[0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]])
    target = torch.tensor([[0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]])

    loss = loss_fn(pred, target)
    expected = 0.0  # 1 - 1^2 = 0
    assert abs(loss.item() - expected) < 1e-5, f"Identical polygons loss: expected {expected}, got {loss.item()}"
    print(f"✓ PolygonIOULoss with identical polygons: {loss.item():.6f}")

    # Test with non-overlapping polygons (loss should be 1 since IoU=0)
    pred = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]])
    target = torch.tensor([[5.0, 5.0, 6.0, 5.0, 6.0, 6.0, 5.0, 6.0]])

    loss = loss_fn(pred, target)
    expected = 1.0  # 1 - 0^2 = 1
    assert abs(loss.item() - expected) < 1e-5, f"Non-overlapping polygons loss: expected {expected}, got {loss.item()}"
    print(f"✓ PolygonIOULoss with non-overlapping polygons: {loss.item():.6f}")


def test_yolox_head_polygon():
    """Test YOLOXHead with polygon mode."""
    from yolox.models import YOLOXHead

    # Create head in standard mode
    head_standard = YOLOXHead(num_classes=80, width=0.5, use_polygon=False)
    assert head_standard.reg_channels == 4, "Standard mode should have 4 reg channels"
    print(f"✓ Standard mode head has {head_standard.reg_channels} regression channels")

    # Create head in polygon mode
    head_polygon = YOLOXHead(num_classes=80, width=0.5, use_polygon=True)
    assert head_polygon.reg_channels == 8, "Polygon mode should have 8 reg channels"
    print(f"✓ Polygon mode head has {head_polygon.reg_channels} regression channels")

    # Test forward pass shapes
    batch_size = 2
    # Simulate input feature maps at different scales
    x8 = torch.randn(batch_size, 128, 80, 80)   # stride 8
    x16 = torch.randn(batch_size, 256, 40, 40)  # stride 16
    x32 = torch.randn(batch_size, 512, 20, 20)  # stride 32

    xin = [x8, x16, x32]

    # Test inference mode (no labels)
    head_polygon.eval()
    with torch.no_grad():
        outputs = head_polygon(xin)

    # Output shape: [batch, num_anchors, 8 + 1 + num_classes]
    expected_channels = 8 + 1 + 80  # polygon coords + obj + classes
    assert outputs.shape[2] == expected_channels, \
        f"Polygon output channels: expected {expected_channels}, got {outputs.shape[2]}"
    print(f"✓ Polygon mode output shape: {outputs.shape}")


def test_backward_compatibility():
    """Test backward compatibility with standard 4-coordinate boxes."""
    from yolox.models import YOLOXHead, IOUloss

    # Standard mode should still use IOUloss
    head = YOLOXHead(num_classes=80, width=0.5, use_polygon=False)
    assert isinstance(head.iou_loss, IOUloss), "Standard mode should use IOUloss"
    print(f"✓ Standard mode uses IOUloss")

    # Test standard box IoU still works
    from yolox.utils import bboxes_iou
    boxes_a = torch.tensor([[100.0, 100.0, 50.0, 50.0]])  # cxcywh format
    boxes_b = torch.tensor([[100.0, 100.0, 50.0, 50.0]])  # same box
    iou = bboxes_iou(boxes_a, boxes_b, xyxy=False)
    assert abs(iou[0, 0].item() - 1.0) < 1e-5, "Identical boxes should have IoU=1"
    print(f"✓ Standard bboxes_iou still works: {iou[0, 0].item():.4f}")


def main():
    print("=" * 60)
    print("YOLOX Polygon Bounding Box Verification Script")
    print("=" * 60)
    print()

    print("1. Testing Polygon Area Computation...")
    test_polygon_area()
    print()

    print("2. Testing Sutherland-Hodgman Polygon Clipping...")
    test_sutherland_hodgman()
    print()

    print("3. Testing Polygon IoU...")
    test_polygon_iou()
    print()

    print("4. Testing Batch Polygon IoU...")
    test_polygon_iou_batch()
    print()

    print("5. Testing Polygon NMS...")
    test_polygon_nms()
    print()

    print("6. Testing PolygonIOULoss...")
    test_polygon_iou_loss()
    print()

    print("7. Testing YOLOXHead with Polygon Mode...")
    test_yolox_head_polygon()
    print()

    print("8. Testing Backward Compatibility...")
    test_backward_compatibility()
    print()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
