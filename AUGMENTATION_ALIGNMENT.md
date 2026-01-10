# Augmentation Alignment Summary

## Overview
This document summarizes the changes made to align all data augmentations with the polygon bounding box support introduced in PR #2.

## Problem Statement
PR #2 added polygon bounding box support to YOLOX, changing from 4-value bounding boxes (x1, y1, x2, y2) to 8-value polygon bounding boxes (x1, y1, x2, y2, x3, y3, x4, y4). However, several data augmentation functions were not updated to handle this new format.

## Format Details

### Traditional Format
- **Label format**: `[x1, y1, x2, y2, class]` (5 columns)
- **Bbox coordinates**: 4 values
- **Use case**: Axis-aligned rectangular bounding boxes

### Polygon Format
- **Label format**: `[x1, y1, x2, y2, x3, y3, x4, y4, class]` (9 columns)
- **Bbox coordinates**: 8 values (4 corner points)
- **Use case**: Arbitrary quadrilateral bounding boxes

## Issues Found and Fixed

### 1. TrainTransform (yolox/data/data_augment.py)
**Issue**: Hardcoded to extract only first 4 coordinates
```python
boxes = targets[:, :4].copy()  # ❌ Always takes 4 values
labels = targets[:, 4].copy()   # ❌ Assumes class at index 4
```

**Fix**: Auto-detect format and handle both
```python
is_polygon = targets.shape[1] > 5
bbox_end_idx = 8 if is_polygon else 4
class_idx = 8 if is_polygon else 4
boxes = targets[:, :bbox_end_idx].copy()
labels = targets[:, class_idx].copy()
```

### 2. MosaicDetection Mosaic Coordinate Transformation
**Issue**: Hardcoded to transform 4 specific coordinates
```python
labels[:, 0] = scale * _labels[:, 0] + padw  # ❌ Only 4 coords
labels[:, 1] = scale * _labels[:, 1] + padh
labels[:, 2] = scale * _labels[:, 2] + padw
labels[:, 3] = scale * _labels[:, 3] + padh
```

**Fix**: Use stride indexing for all coordinates
```python
# Scale and translate all x coordinates (even indices)
labels[:, 0::2] = scale * _labels[:, 0::2] + padw
# Scale and translate all y coordinates (odd indices)
labels[:, 1::2] = scale * _labels[:, 1::2] + padh
```

### 3. MosaicDetection Mosaic Clipping
**Issue**: Hardcoded to clip 4 specific coordinates
```python
np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
# ... repeated for indices 1, 2, 3
```

**Fix**: Use stride indexing
```python
np.clip(mosaic_labels[:, 0::2], 0, 2 * input_w, out=mosaic_labels[:, 0::2])
np.clip(mosaic_labels[:, 1::2], 0, 2 * input_h, out=mosaic_labels[:, 1::2])
```

### 4. MosaicDetection Mixup Flip Operation
**Issue**: Incorrectly extracted and mirrored coordinates
```python
cp_bboxes_origin_np = adjust_box_anns(
    cp_labels[:, :4].copy(), ...  # ❌ Always takes 4 values
)
if FLIP:
    cp_bboxes_origin_np[:, 0::2] = (
        origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]  # ❌ Wrong reversal
    )
```

**Fix**: Auto-detect format and mirror correctly
```python
is_polygon = cp_labels.shape[1] > 5
bbox_end_idx = 8 if is_polygon else 4
cp_bboxes_origin_np = adjust_box_anns(
    cp_labels[:, :bbox_end_idx].copy(), ...
)
if FLIP:
    # Mirror all x coordinates (no reversal)
    cp_bboxes_origin_np[:, 0::2] = origin_w - cp_bboxes_origin_np[:, 0::2]
```

## Already Working

### apply_affine_to_bboxes
✅ Already fixed in PR #2 - detects polygon format and handles both

### _mirror
✅ Already fixed in PR #2 - uses `::2` indexing which works for both formats

### adjust_box_anns
✅ Already working correctly - uses `::2` and `1::2` indexing natively

## Testing

### New Tests Added (tests/test_augmentations_polygon.py)
11 comprehensive tests covering:
- `test_apply_affine_to_bboxes_traditional`
- `test_apply_affine_to_bboxes_polygon`
- `test_mirror_traditional`
- `test_mirror_polygon`
- `test_adjust_box_anns_traditional`
- `test_adjust_box_anns_polygon`
- `test_random_affine_traditional`
- `test_random_affine_polygon`
- `test_train_transform_traditional`
- `test_train_transform_polygon`
- `test_coordinate_clipping_polygon`

### Test Results
- ✅ All 11 new augmentation tests pass
- ✅ All 14 existing polygon tests pass (from PR #2)
- ✅ Total: 25 tests passing

## Backward Compatibility
All changes maintain full backward compatibility:
- Traditional 4-coordinate bounding boxes continue to work unchanged
- Format detection is automatic based on number of columns
- No breaking changes to existing APIs

## Key Patterns for Future Development

When working with bounding boxes that need to support both formats:

1. **Format Detection**: Check number of columns
```python
is_polygon = array.shape[1] > 5  # or > 4 if class not included
```

2. **Coordinate Indexing**: Use stride indexing
```python
x_coords = boxes[:, 0::2]  # All x coordinates
y_coords = boxes[:, 1::2]  # All y coordinates
```

3. **Transformations**: Apply to all coordinate pairs
```python
boxes[:, 0::2] = transform_x(boxes[:, 0::2])
boxes[:, 1::2] = transform_y(boxes[:, 1::2])
```

## Conclusion
All data augmentations in YOLOX now support polygon bounding boxes while maintaining full backward compatibility with traditional rectangular bounding boxes. The implementation uses automatic format detection and stride indexing patterns that are both efficient and easy to maintain.
