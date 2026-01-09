#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import unittest
import torch

from yolox.models import YOLOXHead
from yolox.models.losses import IOUloss


class TestOrientedBBox(unittest.TestCase):

    def test_standard_bbox_head(self):
        """Test that standard bbox mode works as before (4 params)"""
        num_classes = 80
        head = YOLOXHead(num_classes=num_classes, use_oriented_bbox=False)
        
        # Check that reg_preds output 4 channels
        self.assertEqual(head.reg_preds[0].out_channels, 4)
        
        # Test forward pass
        batch_size = 2
        xin = [
            torch.randn(batch_size, 256, 80, 80),
            torch.randn(batch_size, 512, 40, 40),
            torch.randn(batch_size, 1024, 20, 20),
        ]
        
        head.eval()
        with torch.no_grad():
            outputs = head(xin)
        
        # Output should have shape [batch, n_anchors, 4+1+num_classes]
        # where 4 is bbox, 1 is objectness, num_classes is classes
        expected_channels = 4 + 1 + num_classes
        self.assertEqual(outputs.shape[-1], expected_channels)

    def test_oriented_bbox_head(self):
        """Test that oriented bbox mode works with 5 params (cx, cy, w, h, angle)"""
        num_classes = 80
        head = YOLOXHead(num_classes=num_classes, use_oriented_bbox=True)
        
        # Check that reg_preds output 5 channels
        self.assertEqual(head.reg_preds[0].out_channels, 5)
        
        # Test forward pass
        batch_size = 2
        xin = [
            torch.randn(batch_size, 256, 80, 80),
            torch.randn(batch_size, 512, 40, 40),
            torch.randn(batch_size, 1024, 20, 20),
        ]
        
        head.eval()
        with torch.no_grad():
            outputs = head(xin)
        
        # Output should have shape [batch, n_anchors, 5+1+num_classes]
        # where 5 is bbox+angle, 1 is objectness, num_classes is classes
        expected_channels = 5 + 1 + num_classes
        self.assertEqual(outputs.shape[-1], expected_channels)

    def test_oriented_bbox_training(self):
        """Test training mode with oriented bboxes"""
        num_classes = 80
        head = YOLOXHead(num_classes=num_classes, use_oriented_bbox=True)
        head.train()
        
        batch_size = 2
        xin = [
            torch.randn(batch_size, 256, 80, 80),
            torch.randn(batch_size, 512, 40, 40),
            torch.randn(batch_size, 1024, 20, 20),
        ]
        
        # Create dummy labels with 5 bbox params (class, cx, cy, w, h, angle)
        max_labels = 10
        labels = torch.zeros(batch_size, max_labels, 6)
        # Add one object to first batch
        labels[0, 0, :] = torch.tensor([1.0, 320.0, 320.0, 50.0, 50.0, 0.5])
        
        imgs = torch.randn(batch_size, 3, 640, 640)
        
        loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = head(xin, labels, imgs)
        
        # Check that losses are computed
        self.assertIsInstance(loss.item(), float)
        self.assertIsInstance(iou_loss.item(), float)
        self.assertIsInstance(conf_loss.item(), float)
        self.assertIsInstance(cls_loss.item(), float)

    def test_iou_loss_standard(self):
        """Test IOUloss with standard bboxes"""
        loss_fn = IOUloss(reduction="none", use_oriented_bbox=False)
        
        pred = torch.tensor([[100.0, 100.0, 50.0, 50.0]])
        target = torch.tensor([[100.0, 100.0, 50.0, 50.0]])
        
        loss = loss_fn(pred, target)
        
        # Perfect overlap should give loss close to 0
        self.assertLess(loss.item(), 0.1)

    def test_iou_loss_oriented(self):
        """Test IOUloss with oriented bboxes (ignoring angle for IoU)"""
        loss_fn = IOUloss(reduction="none", use_oriented_bbox=True)
        
        # 5-param boxes: cx, cy, w, h, angle
        pred = torch.tensor([[100.0, 100.0, 50.0, 50.0, 0.5]])
        target = torch.tensor([[100.0, 100.0, 50.0, 50.0, 0.7]])
        
        loss = loss_fn(pred, target)
        
        # Perfect overlap in first 4 params should give loss close to 0
        # (angle is not used for IoU calculation)
        self.assertLess(loss.item(), 0.1)

    def test_get_output_and_grid_oriented(self):
        """Test get_output_and_grid handles oriented boxes correctly"""
        head = YOLOXHead(num_classes=80, use_oriented_bbox=True)
        
        batch_size = 2
        # For oriented bbox: 5 (bbox+angle) + 1 (obj) + 80 (classes) = 86 channels
        n_ch = 86
        h, w = 20, 20
        output = torch.randn(batch_size, n_ch, h, w)
        
        processed_output, grid = head.get_output_and_grid(output, 0, stride=32, dtype=torch.float32)
        
        # Should output [batch_size, h*w, 86]
        self.assertEqual(processed_output.shape, (batch_size, h*w, n_ch))
        
        # Check that angle is normalized to [-pi, pi]
        angles = processed_output[..., 4]
        self.assertTrue(torch.all(angles >= -3.15))  # approximately -pi
        self.assertTrue(torch.all(angles <= 3.15))   # approximately pi


if __name__ == "__main__":
    unittest.main()
