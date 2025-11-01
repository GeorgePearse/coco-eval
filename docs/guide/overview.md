# Overview

`coco-eval` is a high-performance Rust library for evaluating object detection models using COCO-style metrics.

## What is COCO Evaluation?

COCO (Common Objects in Context) is a popular benchmark for object detection. The evaluation methodology includes:

- **IoU (Intersection over Union)** - Measures bounding box overlap
- **Precision** - Ratio of correct predictions to all predictions
- **Recall** - Ratio of detected objects to all ground truth objects
- **F-score** - Harmonic mean of precision and recall

## Features

### Core Evaluation

- IoU-based matching between predictions and ground truth
- Configurable IoU thresholds (default: 0.5)
- Per-class and aggregate metrics
- True positives, false positives, false negatives

### Threshold Optimization

- Grid search over confidence thresholds
- Per-class optimization
- F-beta score customization (adjust precision/recall trade-off)

### Advanced Features

- Non-maximum suppression (NMS)
- Confusion matrix generation
- Camera-wise statistics
- Class grouping support

## Architecture

The library is designed for performance:

- **Polars** for DataFrame operations
- **ndarray** for numerical computations
- **Rayon** for parallel processing
- **Zero-copy** where possible

## Use Cases

- **Model Evaluation** - Assess object detection model performance
- **Threshold Tuning** - Find optimal confidence thresholds
- **A/B Testing** - Compare different model versions
- **Continuous Monitoring** - Track model performance over time

## Next Steps

- [API Reference](api.md) - Detailed API documentation
- [Examples](examples.md) - Code examples and use cases
