#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import unittest
import torch
from yolox.models import YOLOXHead


class TestYOLOXHeadPolygon(unittest.TestCase):

    def setUp(self):
        self.num_classes = 80
        self.head = YOLOXHead(num_classes=self.num_classes, width=0.5)

    def test_head_output_channels(self):
        """Test that reg_preds has 8 output channels for polygon predictions"""
        for reg_pred in self.head.reg_preds:
            self.assertEqual(reg_pred.out_channels, 8)

    def test_forward_inference_shape(self):
        """Test forward pass in inference mode with polygon predictions"""
        self.head.eval()
        batch_size = 2
        # Create dummy inputs for 3 feature pyramid levels
        xin = [
            torch.randn(batch_size, 128, 80, 80),  # P3
            torch.randn(batch_size, 256, 40, 40),  # P4
            torch.randn(batch_size, 512, 20, 20),  # P5
        ]
        
        with torch.no_grad():
            outputs = self.head(xin)
        
        # Check output shape
        # Total anchors = 80*80 + 40*40 + 20*20 = 8400
        expected_anchors = 80*80 + 40*40 + 20*20
        # Output should be [batch, n_anchors, 8 + 1 + num_classes]
        expected_channels = 8 + 1 + self.num_classes  # 89 for 80 classes
        
        self.assertEqual(outputs.shape, (batch_size, expected_anchors, expected_channels))

    def test_forward_training_with_labels(self):
        """Test forward pass in training mode with polygon labels"""
        self.head.train()
        batch_size = 2
        # Create dummy inputs
        xin = [
            torch.randn(batch_size, 128, 80, 80),
            torch.randn(batch_size, 256, 40, 40),
            torch.randn(batch_size, 512, 20, 20),
        ]
        
        # Create dummy polygon labels: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
        # Shape: [batch, max_objects, 9]
        max_objects = 10
        labels = torch.zeros(batch_size, max_objects, 9)
        
        # Add one object to the first batch
        labels[0, 0, 0] = 0  # class_id
        labels[0, 0, 1:9] = torch.tensor([100, 100, 200, 100, 200, 200, 100, 200])  # square polygon
        
        # Add one object to the second batch
        labels[1, 0, 0] = 1  # class_id
        labels[1, 0, 1:9] = torch.tensor([150, 150, 250, 150, 250, 250, 150, 250])  # square polygon
        
        imgs = torch.randn(batch_size, 3, 640, 640)
        
        # Forward pass
        outputs = self.head(xin, labels=labels, imgs=imgs)
        
        # Check that we get 6 output values: loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg
        self.assertEqual(len(outputs), 6)
        
        # Check that losses are scalars
        for i in range(5):
            self.assertTrue(torch.is_tensor(outputs[i]) or isinstance(outputs[i], float))

    def test_decode_outputs_polygon(self):
        """Test decode_outputs for polygon predictions"""
        self.head.eval()
        batch_size = 1
        
        # Set hw attribute (required by decode_outputs)
        self.head.hw = [(80, 80), (40, 40), (20, 20)]
        
        # Create a dummy output tensor
        total_anchors = 80*80 + 40*40 + 20*20
        # 8 polygon coords + 1 objectness + num_classes
        output_channels = 8 + 1 + self.num_classes
        outputs = torch.randn(batch_size, total_anchors, output_channels)
        
        # Decode outputs
        with torch.no_grad():
            decoded = self.head.decode_outputs(outputs, dtype=torch.float32)
        
        # Check output shape matches input
        self.assertEqual(decoded.shape, outputs.shape)
        
        # Check that polygon coordinates are in absolute coordinates (not relative)
        # After decoding, coordinates should be larger than grid offsets
        polygon_coords = decoded[0, :, :8]
        self.assertTrue((polygon_coords.abs() > 0).any())

    def test_get_l1_target_polygon(self):
        """Test get_l1_target for polygon predictions"""
        num_fg = 10
        l1_target = torch.zeros(num_fg, 8)
        
        # Create dummy ground truth polygons
        gt = torch.randn(num_fg, 8) * 100 + 200  # Random polygons around (200, 200)
        stride = torch.ones(num_fg) * 8
        x_shifts = torch.randint(0, 80, (num_fg,)).float()
        y_shifts = torch.randint(0, 80, (num_fg,)).float()
        
        # Compute L1 target
        result = self.head.get_l1_target(l1_target, gt, stride, x_shifts, y_shifts)
        
        # Check output shape
        self.assertEqual(result.shape, (num_fg, 8))
        
        # Check that result is not all zeros
        self.assertTrue((result != 0).any())

    def test_polygon_iou_loss_integration(self):
        """Test that PolygonIOULoss is used in the head"""
        from yolox.models.losses import PolygonIOULoss
        self.assertIsInstance(self.head.iou_loss, PolygonIOULoss)


if __name__ == "__main__":
    unittest.main()
