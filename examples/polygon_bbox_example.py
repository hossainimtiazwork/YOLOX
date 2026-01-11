#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Example script demonstrating polygon bounding box support in YOLOX.
This script shows how to use the modified YOLOX model with polygon annotations.
"""

import torch
import numpy as np
import cv2
from yolox.models.yolo_head import YOLOXHead
from yolox.utils import postprocess, vis


def create_sample_polygon_data():
    """
    Create sample polygon bounding box data for demonstration.
    
    Returns:
        labels: numpy array of shape [num_objects, 9]
                Format: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
    """
    # Example: Two rotated rectangles
    labels = np.array([
        # First object: slightly rotated square
        [0, 100.0, 100.0, 200.0, 110.0, 190.0, 210.0, 90.0, 200.0],
        
        # Second object: more rotated rectangle
        [1, 300.0, 300.0, 450.0, 320.0, 440.0, 380.0, 290.0, 360.0],
    ], dtype=np.float32)
    
    return labels


def create_sample_image():
    """
    Create a sample image for demonstration.
    
    Returns:
        image: numpy array of shape [H, W, 3]
    """
    image = np.ones((640, 640, 3), dtype=np.uint8) * 114  # Gray background
    return image


def demo_polygon_format():
    """Demonstrate the polygon bounding box format."""
    print("=" * 60)
    print("Polygon Bounding Box Format Demo")
    print("=" * 60)
    
    # Create sample data
    labels = create_sample_polygon_data()
    image = create_sample_image()
    
    print("\n1. Sample Polygon Labels:")
    print("   Format: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]")
    print(f"   Shape: {labels.shape}")
    print(f"   Data:\n{labels}")
    
    # Visualize polygons on image
    print("\n2. Visualizing Polygons:")
    vis_image = image.copy()
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, label in enumerate(labels):
        class_id = int(label[0])
        points = label[1:].reshape(4, 2).astype(np.int32)
        
        # Draw polygon
        cv2.polylines(vis_image, [points], True, colors[class_id], 2)
        
        # Add label
        text = f"Class {class_id}"
        cv2.putText(vis_image, text, tuple(points[0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)
        
        print(f"   Object {i}: Class {class_id}")
        print(f"   Points: {points}")
    
    print("\n   Image with polygons saved to: polygon_demo.png")
    cv2.imwrite("polygon_demo.png", vis_image)
    
    return labels, image


def demo_model_inference():
    """Demonstrate model inference with polygon format."""
    print("\n" + "=" * 60)
    print("Model Inference Demo")
    print("=" * 60)
    
    # Create YOLOX head
    num_classes = 80
    head = YOLOXHead(num_classes=num_classes)
    head.eval()
    
    print(f"\n1. Model Configuration:")
    print(f"   Number of classes: {num_classes}")
    print(f"   Output format: 8 coords + 1 obj + {num_classes} classes = {8+1+num_classes} channels")
    
    # Create dummy input
    batch_size = 1
    dummy_inputs = [
        torch.randn(batch_size, 256, 80, 80),
        torch.randn(batch_size, 512, 40, 40),
        torch.randn(batch_size, 1024, 20, 20),
    ]
    
    print(f"\n2. Input Feature Maps:")
    for i, x in enumerate(dummy_inputs):
        print(f"   Level {i}: {x.shape}")
    
    # Run inference
    with torch.no_grad():
        outputs = head(dummy_inputs)
    
    print(f"\n3. Model Output:")
    print(f"   Shape: {outputs.shape}")
    print(f"   Format: [batch, n_anchors, channels]")
    print(f"   - Coords (0-7): Polygon vertices (x1,y1,x2,y2,x3,y3,x4,y4)")
    print(f"   - Obj (8): Objectness score")
    print(f"   - Classes (9-88): Class scores")
    
    # Post-process
    print(f"\n4. Post-processing:")
    processed = postprocess(outputs, num_classes, conf_thre=0.3, nms_thre=0.5)
    
    if processed[0] is not None:
        print(f"   Detections: {len(processed[0])}")
        print(f"   Detection format: [8 coords, obj_conf, class_conf, class_pred]")
        print(f"   Detection shape: {processed[0].shape}")
    else:
        print(f"   No detections (expected for random data)")


def demo_data_format_conversion():
    """Demonstrate data format conversion utilities."""
    print("\n" + "=" * 60)
    print("Data Format Conversion Demo")
    print("=" * 60)
    
    # Traditional bbox format (axis-aligned)
    print("\n1. Traditional BBox Format:")
    print("   [x_min, y_min, x_max, y_max]")
    trad_bbox = np.array([100, 100, 200, 200])
    print(f"   Example: {trad_bbox}")
    
    # Polygon format (can be rotated)
    print("\n2. Polygon Format:")
    print("   [x1, y1, x2, y2, x3, y3, x4, y4]")
    
    # Convert traditional bbox to polygon (non-rotated)
    x1, y1, x2, y2 = trad_bbox
    polygon = np.array([x1, y1, x2, y1, x2, y2, x1, y2])
    print(f"   Example (from bbox): {polygon}")
    
    # Rotated polygon
    rotated_polygon = np.array([105, 95, 205, 100, 200, 205, 95, 200])
    print(f"   Example (rotated): {rotated_polygon}")
    
    print("\n3. Advantages of Polygon Format:")
    print("   - Can represent rotated objects")
    print("   - More accurate for non-axis-aligned objects")
    print("   - Useful for text detection, aerial imagery, etc.")
    print("   - Backward compatible with traditional bbox format")


def main():
    """Run all demos."""
    print("\nYOLOX Polygon Bounding Box Support - Example Usage\n")
    
    # Demo 1: Polygon format basics
    labels, image = demo_polygon_format()
    
    # Demo 2: Model inference
    demo_model_inference()
    
    # Demo 3: Data format conversion
    demo_data_format_conversion()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nFor more information, see POLYGON_BBOX_README.md")
    print()


if __name__ == "__main__":
    main()
