# coco-eval

**High-performance Rust library for COCO object detection evaluation metrics with confidence thresholding**

[![Crates.io](https://img.shields.io/crates/v/coco-eval.svg)](https://crates.io/crates/coco-eval)
[![Documentation](https://docs.rs/coco-eval/badge.svg)](https://docs.rs/coco-eval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/GeorgePearse/coco-eval/workflows/CI/badge.svg)](https://github.com/GeorgePearse/coco-eval/actions)

## Overview

`coco-eval` is a comprehensive Rust implementation of COCO (Common Objects in Context) evaluation metrics for object detection. Built for performance and accuracy, it provides all standard COCO metrics with full support for multi-class, multi-image evaluation scenarios.

## ✨ Key Features

### 🎯 Standard COCO Metrics
- **mAP** (mean Average Precision) across all classes and IoU thresholds
- **AP50** (Average Precision at IoU=0.50)
- **AP75** (Average Precision at IoU=0.75)
- **Per-class AP** with independent evaluation sweeps
- **Precision, Recall, F1** at configurable confidence thresholds

### ⚡ Performance
- Written in pure Rust for maximum performance
- Efficient IoU calculation with minimal allocations
- Parallel-friendly design
- Zero-copy operations where possible

### 🔬 COCO-Compliant
- **101-point interpolation** for AP calculation
- **Independent per-class evaluation** (proper COCO methodology)
- **Greedy matching** algorithm (highest confidence first)
- **Standard IoU thresholds** (0.5:0.05:0.95)

### 🎨 Developer-Friendly
- Type-safe COCO format structures
- Comprehensive error handling
- Well-documented API with examples
- **99 comprehensive tests** covering all functionality

## 📊 What Gets Evaluated

```rust
// Input: Ground truth + Predictions
let ground_truth = load_from_file("ground_truth.json")?;
let predictions = load_from_file("predictions.json")?;

// Output: Comprehensive metrics
let metrics = evaluate(&ground_truth, &predictions, None, None)?;

println!("mAP: {:.4}", metrics.map);        // 0.9234
println!("AP50: {:.4}", metrics.ap50);      // 0.9567
println!("AP75: {:.4}", metrics.ap75);      // 0.8901

// Per-class breakdown
for (category_id, ap) in &metrics.ap_per_class {
    println!("Class {}: AP = {:.4}", category_id, ap);
}

// Precision/Recall/F1 at different thresholds
for (threshold, precision) in &metrics.precision_at_thresholds {
    println!("@ {:.2}: P={:.4}", threshold, precision);
}
```

## 🚀 Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
coco-eval = "0.1"
```

### Basic Usage

```rust
use coco_eval::{load_from_file, evaluator::evaluate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load COCO-format JSON files
    let ground_truth = load_from_file("annotations_gt.json")?;
    let predictions = load_from_file("annotations_pred.json")?;

    // Evaluate with default settings
    // - IoU thresholds: 0.5:0.05:0.95 (COCO standard)
    // - Confidence thresholds: 0.0:0.1:1.0
    let metrics = evaluate(&ground_truth, &predictions, None, None)?;

    // Access results
    println!("Overall Performance:");
    println!("  mAP: {:.2}%", metrics.map * 100.0);
    println!("  AP50: {:.2}%", metrics.ap50 * 100.0);
    println!("  AP75: {:.2}%", metrics.ap75 * 100.0);

    Ok(())
}
```

## 📦 What's Included

- **Loader**: Parse COCO JSON with validation
- **Matching**: Greedy IoU-based prediction matching
- **Metrics**: IoU, AP, mAP, Precision, Recall, F1
- **Thresholding**: Confidence filtering and optimization
- **Types**: Type-safe COCO format structures

## 🎓 Use Cases

- **Model Evaluation**: Benchmark object detection models
- **Threshold Tuning**: Find optimal confidence thresholds
- **Dataset Analysis**: Analyze prediction quality per class
- **CI/CD Integration**: Automated performance testing
- **Research**: COCO metric computation for papers

## 🧪 Thoroughly Tested

**99 comprehensive tests** covering:

- ✅ All core functionality
- ✅ Edge cases (empty datasets, 1000+ annotations)
- ✅ Error handling (20 error scenarios)
- ✅ Stress tests (100 classes, 100 images)
- ✅ COCO compliance validation
- ✅ Mathematical correctness

See [TEST_COVERAGE.md](https://github.com/GeorgePearse/coco-eval/blob/main/TEST_COVERAGE.md) for details.

## 📚 Documentation

- **[Installation Guide](getting-started/installation.md)** - Get started quickly
- **[Quick Start](getting-started/quickstart.md)** - Basic usage tutorial
- **[API Reference](guide/api.md)** - Complete API documentation
- **[Examples](guide/examples.md)** - Real-world usage patterns

## 🤝 Contributing

Contributions welcome! See our [Contributing Guide](development/contributing.md).

## 📄 License

MIT License - see [LICENSE](https://github.com/GeorgePearse/coco-eval/blob/main/LICENSE) for details.

## 🔗 Links

- **[GitHub Repository](https://github.com/GeorgePearse/coco-eval)**
- **[Crates.io](https://crates.io/crates/coco-eval)**
- **[API Docs](https://docs.rs/coco-eval)**
- **[COCO Dataset](https://cocodataset.org/)**
