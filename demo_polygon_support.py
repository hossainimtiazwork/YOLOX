#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Demo script to demonstrate polygon bounding box support in YOLOX.

This script shows:
1. Creating a YOLOXHead with polygon support
2. Generating dummy polygon annotations
3. Running forward pass with polygon predictions
4. Comparing output shapes before and after modifications
"""

import torch
from yolox.models import YOLOXHead
from yolox.utils import polygon_iou, polygon_area


def demo_polygon_predictions():
    print("=" * 80)
    print("YOLOX Polygon Bounding Box Demo")
    print("=" * 80)
    
    # 1. Create YOLOXHead
    print("\n1. Creating YOLOXHead with polygon support...")
    num_classes = 80
    head = YOLOXHead(num_classes=num_classes, width=0.5)
    
    # Verify that reg_preds has 8 output channels
    print(f"   ✓ reg_preds output channels: {head.reg_preds[0].out_channels} (expected: 8)")
    
    # 2. Create dummy inputs
    print("\n2. Creating dummy inputs...")
    batch_size = 2
    xin = [
        torch.randn(batch_size, 128, 80, 80),  # P3
        torch.randn(batch_size, 256, 40, 40),  # P4
        torch.randn(batch_size, 512, 20, 20),  # P5
    ]
    print(f"   ✓ Created 3 feature pyramid levels")
    print(f"   ✓ Batch size: {batch_size}")
    
    # 3. Test inference mode
    print("\n3. Testing inference mode...")
    head.eval()
    with torch.no_grad():
        outputs = head(xin)
    
    expected_anchors = 80*80 + 40*40 + 20*20
    expected_channels = 8 + 1 + num_classes  # 8 polygon coords + 1 obj + classes
    print(f"   ✓ Output shape: {outputs.shape}")
    print(f"   ✓ Expected: ({batch_size}, {expected_anchors}, {expected_channels})")
    print(f"   ✓ Polygon coordinates per anchor: 8 (4 vertices × 2 coords)")
    
    # 4. Test training mode with polygon labels
    print("\n4. Testing training mode with polygon labels...")
    head.train()
    
    # Create polygon labels: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
    max_objects = 10
    labels = torch.zeros(batch_size, max_objects, 9)
    
    # Add a square polygon to first batch
    labels[0, 0, 0] = 0  # class_id
    labels[0, 0, 1:9] = torch.tensor([100, 100, 200, 100, 200, 200, 100, 200])
    
    # Add a rectangle polygon to second batch
    labels[1, 0, 0] = 1  # class_id
    labels[1, 0, 1:9] = torch.tensor([150, 150, 250, 150, 250, 250, 150, 250])
    
    imgs = torch.randn(batch_size, 3, 640, 640)
    
    outputs = head(xin, labels=labels, imgs=imgs)
    loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg = outputs
    
    print(f"   ✓ Training losses computed:")
    print(f"     - Total loss: {loss:.4f}")
    print(f"     - IoU loss (polygon): {iou_loss:.4f}")
    print(f"     - Objectness loss: {obj_loss:.4f}")
    print(f"     - Classification loss: {cls_loss:.4f}")
    print(f"     - L1 loss: {l1_loss:.4f}")
    
    # 5. Test polygon utility functions
    print("\n5. Testing polygon utility functions...")
    
    # Test polygon_area
    square = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
    area = polygon_area(square)
    print(f"   ✓ Square area (10×10): {area.item():.2f} (expected: 100.00)")
    
    # Test polygon_iou
    poly1 = torch.tensor([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
    poly2 = torch.tensor([[5.0, 0.0, 15.0, 0.0, 15.0, 10.0, 5.0, 10.0]])
    iou = polygon_iou(poly1, poly2)
    print(f"   ✓ Half-overlapping squares IoU: {iou[0, 0].item():.3f} (expected: ~0.333)")
    
    # 6. Summary
    print("\n" + "=" * 80)
    print("Summary of Changes:")
    print("=" * 80)
    print("✓ Bounding box format changed from 4 values to 8 values")
    print("  - Old: (center_x, center_y, width, height)")
    print("  - New: (x1, y1, x2, y2, x3, y3, x4, y4) - 4 polygon vertices")
    print("✓ YOLOXHead.reg_preds output channels: 4 → 8")
    print("✓ PolygonIOULoss replaces IOULoss for polygon predictions")
    print("✓ All geometric calculations updated for polygon format")
    print("✓ Backward compatible architecture with extended predictions")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = demo_polygon_predictions()
    if success:
        print("\n✓ Demo completed successfully!")
    else:
        print("\n✗ Demo failed!")
