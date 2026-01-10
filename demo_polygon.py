#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Simple demonstration script showing polygon bounding box detection.
This script demonstrates the changes made to support 4-point polygon bounding boxes.
"""

import torch
import numpy as np
from yolox.models import YOLOXHead
from yolox.utils import polygon2xyxy, xyxy2polygon, postprocess


def demo_polygon_detection():
    """Demonstrate polygon detection with YOLOX"""
    print("=" * 60)
    print("Polygon Bounding Box Detection Demo")
    print("=" * 60)
    
    # Initialize model head
    num_classes = 80
    head = YOLOXHead(num_classes=num_classes)
    head.eval()
    
    print(f"\n1. Model Configuration:")
    print(f"   - Number of classes: {num_classes}")
    print(f"   - Bounding box format: Polygon (4 points, 8 values)")
    print(f"   - Output channels: 8 (bbox) + 1 (objectness) + {num_classes} (classes)")
    
    # Create dummy input (simulating backbone features)
    batch_size = 1
    x = [
        torch.randn(batch_size, 256, 80, 80),   # P3 - small objects
        torch.randn(batch_size, 512, 40, 40),   # P4 - medium objects  
        torch.randn(batch_size, 1024, 20, 20),  # P5 - large objects
    ]
    
    print(f"\n2. Input Features:")
    for i, feat in enumerate(x):
        print(f"   - Level P{i+3}: {feat.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = head(x)
    
    print(f"\n3. Model Output:")
    print(f"   - Shape: {output.shape}")
    print(f"   - [batch_size, num_anchors, channels]")
    print(f"   - Channels breakdown:")
    print(f"     * [0:8]   - Polygon coordinates (x1,y1,x2,y2,x3,y3,x4,y4)")
    print(f"     * [8]     - Objectness score")
    print(f"     * [9:{9+num_classes}] - Class probabilities")
    
    # Extract predictions
    bboxes = output[:, :, :8]
    obj_scores = output[:, :, 8]
    cls_scores = output[:, :, 9:]
    
    print(f"\n4. Predictions:")
    print(f"   - Polygon bboxes shape: {bboxes.shape}")
    print(f"   - Objectness scores shape: {obj_scores.shape}")
    print(f"   - Class scores shape: {cls_scores.shape}")
    
    # Show a sample polygon
    sample_polygon = bboxes[0, 0, :].numpy()
    print(f"\n5. Sample Polygon (first detection):")
    print(f"   - Point 1 (x1, y1): ({sample_polygon[0]:.2f}, {sample_polygon[1]:.2f})")
    print(f"   - Point 2 (x2, y2): ({sample_polygon[2]:.2f}, {sample_polygon[3]:.2f})")
    print(f"   - Point 3 (x3, y3): ({sample_polygon[4]:.2f}, {sample_polygon[5]:.2f})")
    print(f"   - Point 4 (x4, y4): ({sample_polygon[6]:.2f}, {sample_polygon[7]:.2f})")
    
    # Convert polygon to xyxy for comparison
    sample_polygon_tensor = torch.tensor(sample_polygon).unsqueeze(0)
    xyxy = polygon2xyxy(sample_polygon_tensor)
    print(f"\n6. Converted to XYXY format (for comparison):")
    print(f"   - x1={xyxy[0, 0]:.2f}, y1={xyxy[0, 1]:.2f}, x2={xyxy[0, 2]:.2f}, y2={xyxy[0, 3]:.2f}")
    
    # Test postprocess with polygon format
    print(f"\n7. Testing Postprocess (NMS with polygons):")
    predictions = postprocess(output, num_classes, conf_thre=0.01, nms_thre=0.45)
    if predictions[0] is not None:
        print(f"   - Number of detections after NMS: {len(predictions[0])}")
        if len(predictions[0]) > 0:
            det = predictions[0][0]
            print(f"   - First detection shape: {det.shape}")
            print(f"   - Contains: [8 polygon coords + obj_conf + cls_conf + cls_id]")
    else:
        print(f"   - No detections above threshold")
    
    # Show comparison: traditional vs polygon
    print(f"\n8. Format Comparison:")
    print(f"   Traditional YOLOX:")
    print(f"   - Bbox: [cx, cy, w, h] (4 values)")
    print(f"   - Limited to axis-aligned rectangles")
    print(f"   ")
    print(f"   Polygon YOLOX (Current):")
    print(f"   - Bbox: [x1, y1, x2, y2, x3, y3, x4, y4] (8 values)")
    print(f"   - Supports arbitrary quadrilaterals")
    print(f"   - Can represent rotated boxes")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo_polygon_detection()
