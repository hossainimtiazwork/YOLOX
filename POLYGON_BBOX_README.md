# Polygon Bounding Box Support for YOLOX

This document describes the polygon bounding box implementation in YOLOX, which extends the traditional 4-coordinate bounding box format to support 8-coordinate polygon format (4 points).

## Overview

The polygon bounding box format allows for more flexible object detection by representing bounding regions as quadrilaterals instead of axis-aligned rectangles. This is particularly useful for:

- Rotated objects
- Objects at oblique angles
- Text detection in documents
- Oriented objects in aerial imagery

## Format Specification

### Traditional Bounding Box Format
- **4 coordinates**: (cx, cy, w, h) - center x, center y, width, height
- Used in original YOLOX implementation

### Polygon Bounding Box Format
- **8 coordinates**: (x1, y1, x2, y2, x3, y3, x4, y4) - 4 corner points
- Represents a quadrilateral polygon
- Points should be ordered (typically clockwise or counter-clockwise)

## Modified Components

### 1. Model Architecture (`yolox/models/yolo_head.py`)

**Changes:**
- `reg_preds` output channels changed from 4 to 8
- Output tensor shape: `[batch, n_anchors, 8 + 1 + num_classes]`
  - 8 coordinates for polygon
  - 1 for objectness score
  - num_classes for class predictions

**Key Methods:**
- `get_output_and_grid()`: Updated to decode 8 coordinates instead of 4
- `decode_outputs()`: Applies grid offset and stride to all 4 points
- `get_losses()`: Handles 8-coordinate targets
- `get_l1_target()`: Computes L1 targets for all 4 points
- `get_geometry_constraint()`: Uses polygon center (mean of 4 points) for anchor matching

### 2. Loss Functions (`yolox/models/losses.py`)

**New Functions:**
- `polygon_area()`: Calculates polygon area using Shoelace formula
- `polygon_to_bbox()`: Converts polygon to axis-aligned bounding box

**Updated:**
- `IOUloss`: Handles both 4-coordinate and 8-coordinate formats
  - For polygons: Converts to bounding boxes for IoU computation
  - Maintains backward compatibility with traditional bbox format

### 3. Utility Functions (`yolox/utils/boxes.py`)

**Updated:**
- `bboxes_iou()`: Supports polygon format by converting to bounding boxes
- `postprocess()`: Handles polygon outputs during NMS
  - Converts polygons to bboxes for NMS operation
  - Maintains polygon coordinates in final output
  - Output format: `[8 coords, obj_conf, class_conf, class_pred]`

### 4. Data Augmentation (`yolox/data/data_augment.py`)

**Updated Functions:**
- `apply_affine_to_bboxes()`: Directly transforms all 4 polygon points
- `_mirror()`: Mirrors all x-coordinates of polygon points
- `TrainTransform`: Handles both polygon and traditional bbox formats
  - For polygons: Expects format `[class, x1, y1, x2, y2, x3, y3, x4, y4]`
  - Scales polygon coordinates during preprocessing
  - Filters based on polygon size

### 5. Visualization (`yolox/utils/visualize.py`)

**Updated:**
- `vis()`: Draws polygons instead of rectangles for 8-coordinate format
  - Uses `cv2.polylines()` for polygon rendering
  - Falls back to rectangles for traditional format

### 6. Inference (`tools/demo.py`)

**Updated:**
- `Predictor.visual()`: Detects and handles polygon format
  - Extracts 8 coordinates when available
  - Adjusts score and class index extraction

## Usage

### Training Data Format

For polygon support, training data should be formatted as:
```python
# Shape: [num_objects, 9]
# Format: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
labels = np.array([
    [0, 100, 100, 200, 100, 200, 200, 100, 200],  # Object 1: class 0
    [1, 300, 300, 400, 300, 400, 400, 300, 400],  # Object 2: class 1
])
```

### Inference Output

Model outputs will have shape `[batch, n_anchors, 89]` for 80 classes:
- Indices 0-7: Polygon coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
- Index 8: Objectness score
- Indices 9-88: Class scores (80 classes)

After postprocessing:
```python
# Shape: [num_detections, 11]
# Format: [x1, y1, x2, y2, x3, y3, x4, y4, obj_conf, class_conf, class_pred]
detections = outputs[0]
```

## Backward Compatibility

All modifications include checks to support the original 4-coordinate bbox format:
- Loss functions check tensor dimensions
- Data augmentation handles both formats
- Visualization falls back to rectangles
- Postprocessing adapts based on input shape

This ensures existing YOLOX code continues to work without modifications.

## Implementation Notes

### IoU Calculation
For computational efficiency, polygon IoU is approximated using bounding box IoU:
1. Convert polygon to axis-aligned bounding box
2. Compute standard IoU between bounding boxes
3. Use result for loss calculation and NMS

This approximation provides good performance while maintaining reasonable accuracy for most use cases.

### Anchor Matching
The geometry constraint uses the center of the polygon (mean of 4 points) to determine if an anchor is responsible for detecting an object.

### Performance Considerations
- Polygon representation adds ~4 extra coordinates per detection
- Memory usage increases proportionally
- Inference speed remains similar due to efficient tensor operations
- IoU computation using bbox approximation maintains good performance

## Testing

A comprehensive test suite verifies:
1. YOLO head outputs correct dimensions (89 channels)
2. Polygon IoU loss calculation
3. Polygon bboxes_iou function
4. Postprocessing with polygon outputs
5. Data augmentation with polygon coordinates

All tests pass successfully, confirming proper implementation.

## Future Improvements

Potential enhancements:
1. Implement true polygon IoU using Shapely or similar libraries
2. Add support for polygons with more than 4 points
3. Optimize polygon-specific operations with CUDA kernels
4. Add polygon-specific data augmentation (e.g., perspective transforms)
5. Implement polygon-aware NMS algorithm

## References

- Original YOLOX paper: [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)
- Shoelace formula: Used for polygon area calculation
- Oriented object detection: Inspiration for polygon bbox format
