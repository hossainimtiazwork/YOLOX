import torch
import numpy as np
import cv2
import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from yolox.utils import (
    polygon_area_np, polygon_iou, polygon_iou_batch, 
    polygon_nms, postprocess_polygon, vis_polygon
)
from yolox.models import YOLOXHead, PolygonIOULoss

def test_polygon_iou():
    print("\n--- Testing Polygon IoU ---")
    # Unit square
    poly_a = torch.tensor([0, 0, 1, 0, 1, 1, 0, 1], dtype=torch.float32)
    # 0.5 overlap
    poly_b = torch.tensor([0.5, 0, 1.5, 0, 1.5, 1, 0.5, 1], dtype=torch.float32)
    
    iou = polygon_iou(poly_a, poly_b)
    print(f"IoU (0.5 overlap expected): {iou.item():.4f}")
    assert abs(iou.item() - 0.3333) < 1e-3, f"Expected ~0.3333, got {iou.item()}"

    # Full overlap
    iou_full = polygon_iou(poly_a, poly_a)
    print(f"IoU (Full overlap expected): {iou_full.item():.4f}")
    assert abs(iou_full.item() - 1.0) < 1e-3
    
    # Disjoint
    poly_c = torch.tensor([2, 2, 3, 2, 3, 3, 2, 3], dtype=torch.float32)
    iou_disjoint = polygon_iou(poly_a, poly_c)
    print(f"IoU (Disjoint expected): {iou_disjoint.item():.4f}")
    assert iou_disjoint.item() == 0.0
    print("Polygon IoU tests passed!")

def test_polygon_iou_batch():
    print("\n--- Testing Polygon IoU Batch ---")
    gt = torch.tensor([
        [0, 0, 1, 0, 1, 1, 0, 1],
        [2, 2, 3, 2, 3, 3, 2, 3]
    ], dtype=torch.float32)
    pred = torch.tensor([
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0.5, 0, 1.5, 0, 1.5, 1, 0.5, 1],
        [2, 2, 3, 2, 3, 3, 2, 3]
    ], dtype=torch.float32)
    
    ious = polygon_iou_batch(gt, pred)
    print(f"Batch IoUs shape: {ious.shape}")
    print(f"IoUs:\n{ious}")
    assert ious.shape == (2, 3)
    assert abs(ious[0, 0] - 1.0) < 1e-3
    assert abs(ious[1, 2] - 1.0) < 1e-3
    print("Polygon IoU Batch tests passed!")

def test_polygon_nms():
    print("\n--- Testing Polygon NMS ---")
    # Two overlapping polygons
    polygons = torch.tensor([
        [0, 0, 10, 0, 10, 10, 0, 10], # score 0.9
        [1, 1, 11, 1, 11, 11, 1, 11]  # score 0.8, should be suppressed
    ], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    
    # NMS with iou threshold 0.5
    keep = polygon_nms(polygons, scores, 0.5)
    print(f"Keep indices (expected [0]): {keep}")
    assert len(keep) == 1 and keep[0] == 0
    print("Polygon NMS tests passed!")

def test_model_forward():
    print("\n--- Testing YOLOXHead Forward Pass ---")
    num_classes = 20
    head = YOLOXHead(num_classes=num_classes, width=0.5, use_polygon=True)
    
    # Dummy inputs for 3 FPN levels
    # Level 1: stride 8, 640/8 = 80
    # Level 2: stride 16, 640/16 = 40
    # Level 3: stride 32, 640/32 = 20
    xin = [
        torch.zeros(1, int(256 * 0.5), 80, 80),
        torch.zeros(1, int(512 * 0.5), 40, 40),
        torch.zeros(1, int(1024 * 0.5), 20, 20)
    ]
    
    outputs = head(xin)
    print(f"Output shape: {outputs.shape}") # [batch, n_anchors, 8+1+num_classes]
    expected_anchors = 80*80 + 40*40 + 20*20
    assert outputs.shape == (1, expected_anchors, 9 + num_classes)
    print("Model Forward Pass tests passed!")

def test_loss_computation():
    print("\n--- Testing Polygon Loss Computation ---")
    num_classes = 2
    head = YOLOXHead(num_classes=num_classes, width=0.5, use_polygon=True)
    
    # Dummy inputs
    xin = [
        torch.zeros(1, int(256 * 0.5), 8, 8),
        torch.zeros(1, int(512 * 0.5), 4, 4),
        torch.zeros(1, int(1024 * 0.5), 2, 2)
    ]
    
    # Dummy labels: [class, x1, y1, x2, y2, x3, y3, x4, y4]
    labels = torch.zeros(1, 10, 9)
    labels[0, 0] = torch.tensor([0, 10, 10, 20, 10, 20, 20, 10, 20]) # One object
    
    imgs = torch.zeros(1, 3, 64, 64)
    
    # Set to training mode for loss computation
    head.train()
    loss_results = head(xin, labels=labels, imgs=imgs)
    
    loss, loss_iou, loss_obj, loss_cls, loss_l1, proportion = loss_results
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss IoU: {loss_iou.item():.4f}")
    print(f"Loss Obj: {loss_obj.item():.4f}")
    
    assert not torch.isnan(loss)
    print("Polygon Loss tests passed!")

if __name__ == "__main__":
    try:
        test_polygon_iou()
        test_polygon_iou_batch()
        test_polygon_nms()
        test_model_forward()
        test_loss_computation()
        print("\nALL VERIFICATION TESTS PASSED!")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
