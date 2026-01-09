#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Test visualization with polygon bounding boxes
"""

import numpy as np
import cv2
import os
import tempfile
from yolox.utils import vis


def test_polygon_visualization():
    """Test that visualization works with polygon format"""
    print("Testing polygon visualization...")
    
    # Create a blank image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Create some test polygons
    # Format: [x1, y1, x2, y2, x3, y3, x4, y4]
    boxes = np.array([
        # Square polygon
        [100, 100, 200, 100, 200, 200, 100, 200],
        # Rotated polygon (diamond shape)
        [320, 150, 370, 200, 320, 250, 270, 200],
        # Trapezoid
        [400, 300, 500, 300, 520, 400, 380, 400],
    ])
    
    scores = np.array([0.95, 0.87, 0.72])
    cls_ids = np.array([0, 1, 2])
    class_names = ["person", "bicycle", "car"]
    
    print(f"Drawing {len(boxes)} polygon bounding boxes...")
    result = vis(img, boxes, scores, cls_ids, conf=0.5, class_names=class_names)
    
    # Save the visualization
    output_path = os.path.join(tempfile.gettempdir(), "polygon_visualization_test.jpg")
    cv2.imwrite(output_path, result)
    print(f"Visualization saved to: {output_path}")
    
    # Also test with traditional bbox format to ensure backward compatibility
    print("\nTesting backward compatibility with traditional bboxes...")
    img2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Traditional format: [x1, y1, x2, y2]
    boxes_trad = np.array([
        [100, 100, 200, 200],
        [270, 150, 370, 250],
        [380, 300, 520, 400],
    ])
    
    result2 = vis(img2, boxes_trad, scores, cls_ids, conf=0.5, class_names=class_names)
    output_path2 = os.path.join(tempfile.gettempdir(), "traditional_bbox_test.jpg")
    cv2.imwrite(output_path2, result2)
    print(f"Traditional bbox visualization saved to: {output_path2}")
    
    print("\nVisualization test completed successfully!")
    return True


if __name__ == "__main__":
    test_polygon_visualization()
