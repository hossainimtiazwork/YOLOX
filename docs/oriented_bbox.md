# Oriented Bounding Box Support

This document describes how to use oriented bounding boxes (OBB) with YOLOX detection head.

## Overview

The YOLOX detection head has been extended to support oriented bounding boxes. Oriented bounding boxes are represented with 5 parameters instead of the standard 4:

- **Standard bounding box**: `(cx, cy, w, h)` - center x, center y, width, height
- **Oriented bounding box**: `(cx, cy, w, h, angle)` - center x, center y, width, height, rotation angle

The rotation angle is represented in radians and normalized to the range `[-π, π]`.

## Usage

### Enabling Oriented Bounding Boxes

To enable oriented bounding box support, set the `use_oriented_bbox` parameter to `True` when creating a `YOLOXHead`:

```python
from yolox.models import YOLOXHead

# Standard bounding boxes (default)
head_standard = YOLOXHead(num_classes=80, use_oriented_bbox=False)

# Oriented bounding boxes
head_oriented = YOLOXHead(num_classes=80, use_oriented_bbox=True)
```

### Training with Oriented Bounding Boxes

When training with oriented bounding boxes, your labels should have 6 values per object instead of 5:

```python
# Standard bbox labels: [class_id, cx, cy, w, h]
standard_labels = torch.tensor([
    [0, 320.0, 320.0, 50.0, 50.0],  # Object 1
    [1, 400.0, 400.0, 60.0, 40.0],  # Object 2
])

# Oriented bbox labels: [class_id, cx, cy, w, h, angle]
oriented_labels = torch.tensor([
    [0, 320.0, 320.0, 50.0, 50.0, 0.5],   # Object 1 with rotation
    [1, 400.0, 400.0, 60.0, 40.0, -0.3],  # Object 2 with rotation
])
```

The angle should be in radians and will be automatically normalized to `[-π, π]` during training.

### Inference with Oriented Bounding Boxes

During inference, the output will include the rotation angle as the 5th parameter:

```python
import torch
from yolox.models import YOLOXHead

head = YOLOXHead(num_classes=80, use_oriented_bbox=True)
head.eval()

# Prepare input
xin = [
    torch.randn(1, 256, 80, 80),
    torch.randn(1, 512, 40, 40),
    torch.randn(1, 1024, 20, 20),
]

# Run inference
with torch.no_grad():
    outputs = head(xin)

# Output shape: [batch_size, num_anchors, 5 + 1 + num_classes]
# where:
#   - First 5 values: cx, cy, w, h, angle
#   - Next 1 value: objectness score
#   - Last num_classes values: class probabilities

print(f"Output shape: {outputs.shape}")
# Output shape: torch.Size([1, 8400, 86])
# 86 = 5 (bbox+angle) + 1 (objectness) + 80 (classes)
```

## Implementation Details

### Loss Computation

The oriented bounding box implementation uses a hybrid approach for loss computation:

1. **IoU Loss**: Computed using only the first 4 parameters (cx, cy, w, h), ignoring the rotation angle. This provides a reasonable approximation for most cases and is computationally efficient.

2. **L1 Loss**: Computed for all 5 parameters including the rotation angle, ensuring the model learns to predict the correct orientation.

3. **Classification and Objectness Losses**: Remain unchanged from the standard implementation.

### Angle Normalization

The rotation angle is normalized using the `tanh` activation function:

```python
angle = torch.tanh(raw_angle) * π
```

This ensures the angle is always in the range `[-π, π]`.

### Backward Compatibility

The implementation maintains full backward compatibility with the standard bounding box mode:

- When `use_oriented_bbox=False` (default), the behavior is identical to the original YOLOX implementation
- All existing code, models, and datasets continue to work without modification
- The output channel count automatically adjusts based on the mode:
  - Standard mode: `4 + 1 + num_classes` channels
  - Oriented mode: `5 + 1 + num_classes` channels

## Example: Complete Training Setup

```python
import torch
from yolox.models import YOLOX, YOLOXHead
from yolox.models import YOLOPAFPN

# Create model with oriented bounding box support
backbone = YOLOPAFPN()
head = YOLOXHead(num_classes=80, use_oriented_bbox=True)
model = YOLOX(backbone=backbone, head=head)

# Prepare training data
batch_size = 4
images = torch.randn(batch_size, 3, 640, 640)

# Labels with oriented bounding boxes
# Shape: [batch_size, max_objects, 6]
# Format: [class_id, cx, cy, w, h, angle]
max_objects = 50
labels = torch.zeros(batch_size, max_objects, 6)
# Add some sample objects
labels[0, 0, :] = torch.tensor([1.0, 320.0, 320.0, 50.0, 50.0, 0.5])
labels[1, 0, :] = torch.tensor([2.0, 400.0, 400.0, 60.0, 40.0, -0.3])

# Training mode
model.train()
outputs = model(images, targets=labels)

print(f"Total loss: {outputs['total_loss']}")
print(f"IoU loss: {outputs['iou_loss']}")
print(f"Classification loss: {outputs['cls_loss']}")
print(f"Objectness loss: {outputs['conf_loss']}")
print(f"L1 loss: {outputs['l1_loss']}")
```

## Limitations

1. **IoU Calculation**: The current implementation computes IoU using only the axis-aligned bounding box (first 4 parameters). For more accurate rotated IoU calculation, consider implementing specialized rotated IoU algorithms.

2. **Visualization**: The visualization utilities currently visualize only the axis-aligned bounding box. You may need to implement custom visualization for rotated boxes.

3. **Post-processing**: Non-maximum suppression (NMS) is applied using the axis-aligned bounding box. For better results with heavily rotated objects, consider using rotated NMS.

## Future Improvements

Potential enhancements for oriented bounding box support:

- Implement true rotated IoU calculation for more accurate loss computation
- Add rotated NMS for better post-processing
- Add visualization support for rotated bounding boxes
- Support alternative angle representations (e.g., quaternions for 3D boxes)
