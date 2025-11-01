# coco-eval

High-performance Rust library for COCO-style object detection evaluation metrics with confidence threshold optimization.

## Overview

`coco-eval` is a Rust implementation of evaluation metrics for object detection models, inspired by the COCO (Common Objects in Context) evaluation methodology. This library provides:

- **Fast evaluation** of object detection predictions against ground truth
- **Confidence threshold optimization** via grid search
- **IoU-based matching** with configurable thresholds
- **Non-maximum suppression (NMS)** support
- **F-beta score calculation** for precision/recall trade-offs
- **Confusion matrix generation** and visualization
- **Camera-wise statistics** for performance breakdown

## Key Features

### Performance-Focused

Built in Rust for maximum performance, targeting 10x+ speedup over the Python reference implementation.

### COCO-Style Metrics

Follows standard object detection evaluation practices:

- Intersection over Union (IoU) computation
- Configurable IoU thresholds
- Per-class metrics
- Class grouping support

### Threshold Optimization

Automatically finds optimal confidence thresholds for each class through grid search, balancing precision and recall.

### DataFrame Integration

Leverages [Polars](https://pola.rs/) for efficient data processing and analysis.

## Quick Example

```rust
use coco_eval::Evaluator;

fn main() {
    // Create evaluator with configuration
    let evaluator = Evaluator::new()
        .iou_threshold(0.5)
        .confidence_thresholds(vec![0.1, 0.2, 0.3, 0.4, 0.5])
        .build();

    // Load predictions and ground truth
    let predictions = load_predictions("predictions.json");
    let ground_truth = load_ground_truth("ground_truth.json");

    // Evaluate
    let results = evaluator.evaluate(&predictions, &ground_truth)?;

    // Print metrics
    println!("Precision: {:.3}", results.precision);
    println!("Recall: {:.3}", results.recall);
    println!("F1 Score: {:.3}", results.f1_score);
}
```

## Python Reference Implementation

This Rust library is a port of a production Python implementation used at BinIt. The original Python code is preserved in the `references/` directory as a specification and validation reference.

!!! warning "Read-Only Reference"
    The `references/` directory contains the Python implementation for reference only. **Never modify these files.** All development happens in the Rust codebase.

## Performance Goals

Target performance improvements over Python:

- **10x faster** evaluation on typical workloads
- **Zero-copy** operations where possible
- **Parallel processing** for independent computations
- **Memory efficiency** through Rust's ownership model

## Get Started

Ready to use `coco-eval`? Check out the [Installation Guide](getting-started/installation.md) and [Quick Start Tutorial](getting-started/quickstart.md).

## License

MIT License - see [LICENSE](https://github.com/GeorgePearse/coco-eval/blob/main/LICENSE) for details.
