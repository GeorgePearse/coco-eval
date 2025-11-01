# TODO: Future Enhancements

## Segmentation Support

### Mask → Segmentation Transformation
- [ ] Implement mask-to-polygon conversion for instance segmentation
- [ ] Support COCO RLE (Run-Length Encoding) format for compact mask representation
- [ ] Add polygon simplification algorithms (Douglas-Peucker)
- [ ] Implement mask IoU calculation for segmentation metrics
- [ ] Add segmentation-specific metrics (Dice coefficient, boundary IoU)
- [ ] Support panoptic segmentation evaluation

**Reference implementations:**
- `pycocotools` mask utilities
- `supervision` polygon utilities
- OpenCV contour detection

## GPU Acceleration

### Triton-Lang Implementations
- [ ] Investigate Triton-lang for GPU-accelerated IoU computation
- [ ] Implement batched IoU matrix calculation on GPU
- [ ] GPU-accelerated NMS using Triton kernels
- [ ] Parallel threshold optimization on GPU
- [ ] Benchmark GPU vs CPU performance on large datasets

**Potential benefits:**
- 10-100x speedup for large-scale evaluation (1M+ boxes)
- Efficient batch processing for multiple images
- Real-time evaluation for live inference systems

**Considerations:**
- Triton availability and platform support
- CPU fallback for non-GPU systems
- Memory transfer overhead
- Feature flag for optional GPU support

## Additional Features

### Performance Optimizations
- [ ] SIMD optimizations for IoU calculation
- [ ] Parallel processing with Rayon for multi-class evaluation
- [ ] Memory pooling for bbox allocations
- [ ] Cached intermediate results for repeated evaluations

### Metrics
- [ ] Per-size metrics (small, medium, large objects)
- [ ] Cross-class confusion matrix
- [ ] False positive analysis by type (localization vs classification errors)
- [ ] Confidence calibration metrics

### Visualization
- [ ] Generate precision-recall curves
- [ ] Plot F-beta curves for threshold selection
- [ ] Heatmaps for per-class performance
- [ ] Interactive HTML reports

### Utilities
- [ ] Dataset splitting (train/val/test)
- [ ] Annotation format conversion (YOLO, Pascal VOC, etc.)
- [ ] Dataset statistics and validation
- [ ] Augmentation pipeline integration

## Priority

**High Priority** (for 0.2.0 release):
1. GPU acceleration with Triton
2. Segmentation mask support

**Medium Priority** (for 0.3.0 release):
1. Per-size metrics
2. Precision-recall curve generation

**Low Priority** (backlog):
1. Additional format conversions
2. Interactive visualizations

## References

### Testing in Rust
- [Why does Rust's testing tools seem so much less mature than other languages?](https://www.reddit.com/r/rust/comments/wmspjb/why_does_rusts_testing_tools_seem_so_much_less/)
  - Discussion on the state of testing tooling in Rust
  - Context for our comprehensive test suite approach (99 tests)
  - Highlights gaps in property-based testing, snapshot testing, etc.
