#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

"""
Example script demonstrating oriented bounding box support in YOLOX.

This script shows:
1. How to create a YOLOX model with oriented bounding box support
2. How to prepare training data with rotation angles
3. How to run inference and extract oriented bounding boxes
"""

import torch
from yolox.models import YOLOX, YOLOXHead, YOLOPAFPN


def example_standard_bbox():
    """Example using standard bounding boxes (default mode)"""
    print("=" * 60)
    print("Example 1: Standard Bounding Boxes")
    print("=" * 60)
    
    # Create model with standard bounding boxes
    backbone = YOLOPAFPN()
    head = YOLOXHead(num_classes=80, use_oriented_bbox=False)
    model = YOLOX(backbone=backbone, head=head)
    
    print(f"Model created with use_oriented_bbox=False")
    print(f"Regression output channels: {head.reg_preds[0].out_channels}")
    
    # Inference
    model.eval()
    batch_size = 1
    images = torch.randn(batch_size, 3, 640, 640)
    
    with torch.no_grad():
        outputs = model(images)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Output channels: {outputs.shape[-1]}")
    print(f"  - Bounding box: 4 params (cx, cy, w, h)")
    print(f"  - Objectness: 1 param")
    print(f"  - Classes: 80 params")
    print(f"  - Total: 4 + 1 + 80 = 85")
    print()


def example_oriented_bbox():
    """Example using oriented bounding boxes"""
    print("=" * 60)
    print("Example 2: Oriented Bounding Boxes")
    print("=" * 60)
    
    # Create model with oriented bounding boxes
    backbone = YOLOPAFPN()
    head = YOLOXHead(num_classes=80, use_oriented_bbox=True)
    model = YOLOX(backbone=backbone, head=head)
    
    print(f"Model created with use_oriented_bbox=True")
    print(f"Regression output channels: {head.reg_preds[0].out_channels}")
    
    # Inference
    model.eval()
    batch_size = 1
    images = torch.randn(batch_size, 3, 640, 640)
    
    with torch.no_grad():
        outputs = model(images)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Output channels: {outputs.shape[-1]}")
    print(f"  - Bounding box: 5 params (cx, cy, w, h, angle)")
    print(f"  - Objectness: 1 param")
    print(f"  - Classes: 80 params")
    print(f"  - Total: 5 + 1 + 80 = 86")
    
    # Extract first detection (for demonstration)
    first_detection = outputs[0, 0, :]
    cx, cy, w, h, angle = first_detection[:5]
    objectness = first_detection[5]
    class_scores = first_detection[6:]
    
    print(f"\nFirst detection:")
    print(f"  - Center: ({cx:.2f}, {cy:.2f})")
    print(f"  - Size: {w:.2f} x {h:.2f}")
    print(f"  - Angle: {angle:.4f} radians ({angle * 180 / 3.14159:.2f} degrees)")
    print(f"  - Objectness: {objectness:.4f}")
    print(f"  - Top class score: {class_scores.max():.4f}")
    print()


def example_training():
    """Example showing training with oriented bounding boxes"""
    print("=" * 60)
    print("Example 3: Training with Oriented Bounding Boxes")
    print("=" * 60)
    
    # Create model
    backbone = YOLOPAFPN()
    head = YOLOXHead(num_classes=80, use_oriented_bbox=True)
    model = YOLOX(backbone=backbone, head=head)
    
    # Training mode
    model.train()
    
    # Prepare training data
    batch_size = 2
    images = torch.randn(batch_size, 3, 640, 640)
    
    # Labels with oriented bounding boxes
    # Shape: [batch_size, max_objects, 6]
    # Format: [class_id, cx, cy, w, h, angle]
    max_objects = 50
    labels = torch.zeros(batch_size, max_objects, 6)
    
    # Add some sample objects
    # Object 1: class 0, center (320, 320), size 50x50, rotated 0.5 radians (~28.6 degrees)
    labels[0, 0, :] = torch.tensor([0.0, 320.0, 320.0, 50.0, 50.0, 0.5])
    # Object 2: class 1, center (400, 400), size 60x40, rotated -0.3 radians (~-17.2 degrees)
    labels[1, 0, :] = torch.tensor([1.0, 400.0, 400.0, 60.0, 40.0, -0.3])
    
    print(f"Training batch:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image shape: {images.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Sample label 1: class={labels[0, 0, 0]}, "
          f"center=({labels[0, 0, 1]}, {labels[0, 0, 2]}), "
          f"size=({labels[0, 0, 3]}x{labels[0, 0, 4]}), "
          f"angle={labels[0, 0, 5]:.3f} rad")
    
    # Forward pass
    outputs = model(images, targets=labels)
    
    print(f"\nLoss values:")
    print(f"  - Total loss: {outputs['total_loss']:.4f}")
    print(f"  - IoU loss: {outputs['iou_loss']:.4f}")
    print(f"  - Classification loss: {outputs['cls_loss']:.4f}")
    print(f"  - Objectness loss: {outputs['conf_loss']:.4f}")
    print(f"  - L1 loss: {outputs['l1_loss']:.4f}")
    print()


if __name__ == "__main__":
    # Run examples
    example_standard_bbox()
    example_oriented_bbox()
    example_training()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nFor more details, see docs/oriented_bbox.md")
