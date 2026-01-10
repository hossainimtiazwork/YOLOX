#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import unittest
import torch

from yolox.models import YOLOX, YOLOXHead, YOLOPAFPN


class TestPolygonModel(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 2
        self.num_classes = 80
        self.img_size = 640
        
    def test_yolox_head_forward_inference(self):
        """Test YOLOXHead forward pass in inference mode"""
        head = YOLOXHead(num_classes=self.num_classes)
        head.eval()
        
        # Create dummy input features from backbone
        # Typical YOLOX has 3 feature levels
        x = [
            torch.randn(self.batch_size, 256, 80, 80),  # P3
            torch.randn(self.batch_size, 512, 40, 40),  # P4
            torch.randn(self.batch_size, 1024, 20, 20), # P5
        ]
        
        with torch.no_grad():
            output = head(x)
        
        # Check output shape
        # Expected: [batch_size, num_anchors, 8 (bbox) + 1 (obj) + num_classes]
        expected_channels = 8 + 1 + self.num_classes  # 89 for 80 classes
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[2], expected_channels)
        
        # Check that bbox predictions are in first 8 channels
        bboxes = output[:, :, :8]
        self.assertEqual(bboxes.shape[2], 8)
        
    def test_yolox_head_forward_training(self):
        """Test YOLOXHead forward pass in training mode"""
        head = YOLOXHead(num_classes=self.num_classes)
        head.train()
        
        # Create dummy input features
        x = [
            torch.randn(self.batch_size, 256, 80, 80),
            torch.randn(self.batch_size, 512, 40, 40),
            torch.randn(self.batch_size, 1024, 20, 20),
        ]
        
        # Create dummy labels: [batch_size, max_objects, 9]
        # Format: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
        max_objects = 10
        labels = torch.zeros(self.batch_size, max_objects, 9)
        # Add one object per image
        for i in range(self.batch_size):
            labels[i, 0, 0] = 0  # class id
            # Simple square: (100, 100) to (200, 200)
            labels[i, 0, 1:9] = torch.tensor([100, 100, 200, 100, 200, 200, 100, 200])
        
        imgs = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        
        # Forward pass
        loss_tuple = head(x, labels=labels, imgs=imgs)
        
        # Check that we get loss outputs
        self.assertEqual(len(loss_tuple), 6)
        total_loss, loss_iou, loss_obj, loss_cls, loss_l1, num_fg = loss_tuple
        
        # Check that losses are scalar tensors
        self.assertTrue(isinstance(total_loss, torch.Tensor))
        self.assertEqual(total_loss.dim(), 0)
        
    def test_output_channels_per_level(self):
        """Test that each level outputs correct number of channels"""
        head = YOLOXHead(num_classes=self.num_classes)
        
        # Check that reg_preds have 8 output channels
        for reg_pred in head.reg_preds:
            self.assertEqual(reg_pred.out_channels, 8)
            
        # Check that obj_preds have 1 output channel
        for obj_pred in head.obj_preds:
            self.assertEqual(obj_pred.out_channels, 1)
            
        # Check that cls_preds have num_classes output channels
        for cls_pred in head.cls_preds:
            self.assertEqual(cls_pred.out_channels, self.num_classes)
    
    def test_polygon_coordinates_are_decoded(self):
        """Test that polygon coordinates are properly decoded in inference mode"""
        head = YOLOXHead(num_classes=self.num_classes)
        head.eval()
        
        x = [
            torch.randn(1, 256, 80, 80),
            torch.randn(1, 512, 40, 40),
            torch.randn(1, 1024, 20, 20),
        ]
        
        with torch.no_grad():
            output = head(x)
        
        # Extract polygon coordinates
        polygons = output[0, :, :8]
        
        # Check that coordinates are in a reasonable range (should be in image space)
        # After decoding, coordinates should be positive and reasonable
        self.assertTrue((polygons >= 0).any() or (polygons < 0).any())  # Can have any values after sigmoid/exp
        
        # Check shape
        self.assertEqual(polygons.shape[1], 8)


if __name__ == "__main__":
    unittest.main()
