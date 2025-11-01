# coco-eval

A Rust library for COCO (Common Objects in Context) object detection evaluation metrics with confidence thresholding support.

[![Crates.io](https://img.shields.io/crates/v/coco-eval.svg)](https://crates.io/crates/coco-eval)
[![Documentation](https://docs.rs/coco-eval/badge.svg)](https://docs.rs/coco-eval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Standard COCO Metrics**
  - mAP (mean Average Precision)
  - AP50 (Average Precision at IoU=0.50)
  - AP75 (Average Precision at IoU=0.75)

- **Confidence Thresholding**
  - Precision at various confidence thresholds
  - Recall at various confidence thresholds
  - F1 Score at various confidence thresholds
  - Optimal threshold finding

- **Core Functionality**
  - Load and parse COCO format JSON annotations
  - Calculate IoU (Intersection over Union) between bounding boxes
  - Filter predictions by confidence scores
  - Evaluate at multiple confidence and IoU thresholds

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
coco-eval = "0.1"
```

## Quick Start

```rust
use coco_eval::{load_from_file, evaluator::evaluate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load ground truth and predictions
    let ground_truth = load_from_file("ground_truth.json")?;
    let predictions = load_from_file("predictions.json")?;

    // Evaluate with default thresholds
    let metrics = evaluate(&ground_truth, &predictions, None, None)?;

    // Print results
    println!("mAP: {:.4}", metrics.map);
    println!("AP50: {:.4}", metrics.ap50);
    println!("AP75: {:.4}", metrics.ap75);

    // Print precision/recall/F1 at different confidence thresholds
    for (threshold, precision) in &metrics.precision_at_thresholds {
        println!("Precision @ {:.2}: {:.4}", threshold, precision);
    }

    Ok(())
}
```

## Usage Examples

### Loading COCO Annotations

```rust
use coco_eval::load_from_file;

// Load from file
let dataset = load_from_file("annotations.json")?;

// Or from a JSON string
use coco_eval::load_from_string;
let json = r#"{
    "annotations": [...],
    "categories": [...]
}"#;
let dataset = load_from_string(json)?;
```

### Confidence Thresholding

```rust
use coco_eval::{filter_dataset_by_confidence, generate_threshold_range};

// Filter predictions with confidence >= 0.5
let filtered = filter_dataset_by_confidence(&predictions, 0.5)?;

// Generate threshold range for evaluation
let thresholds = generate_threshold_range(0.0, 1.0, 11)?;
// Returns: [0.0, 0.1, 0.2, ..., 1.0]
```

### Calculating IoU

```rust
use coco_eval::BoundingBox;
use coco_eval::metrics::iou::calculate_iou;

let bbox1 = BoundingBox::new(10.0, 10.0, 50.0, 50.0);
let bbox2 = BoundingBox::new(30.0, 30.0, 50.0, 50.0);

let iou = calculate_iou(&bbox1, &bbox2);
println!("IoU: {:.4}", iou);
```

### Computing Precision, Recall, and F1

```rust
use coco_eval::metrics::{precision_recall, f1_score};

// From TP, FP, FN counts
let pr = precision_recall::calculate_precision_recall(80, 20, 15);
println!("Precision: {:.4}", pr.precision);
println!("Recall: {:.4}", pr.recall);

let f1 = f1_score::calculate_f1_from_pr(&pr);
println!("F1 Score: {:.4}", f1);
```

## COCO Format

This library expects annotations in the standard COCO JSON format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1200.0,
      "score": 0.95
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "human"
    }
  ]
}
```

**Bounding Box Format**: `[x, y, width, height]` (LTWH - Left, Top, Width, Height)

**Score Field**: Optional for ground truth, required for predictions

## Metrics Explained

### Average Precision (AP)

AP summarizes the precision-recall curve as the weighted mean of precisions achieved at each threshold:

```
AP = Σ(Rₙ - Rₙ₋₁) × Pₙ
```

### AP50 and AP75

- **AP50**: Average Precision at IoU threshold of 0.50 (lenient matching)
- **AP75**: Average Precision at IoU threshold of 0.75 (strict matching)

### mAP (mean Average Precision)

The mean of AP values across all categories and IoU thresholds (0.5:0.05:0.95 for COCO).

### Precision

```
Precision = TP / (TP + FP)
```

Measures how many selected items are relevant.

### Recall

```
Recall = TP / (TP + FN)
```

Measures how many relevant items are selected.

### F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Harmonic mean of precision and recall.

## Examples

See the `examples/` directory for complete working examples:

```bash
# Run basic evaluation example
cargo run --example basic_evaluation

# Run threshold optimization example
cargo run --example threshold_optimization
```

## Development

### Building

```bash
cargo build --release
```

### Testing

```bash
cargo test
```

### Documentation

```bash
cargo doc --open
```

## Roadmap

- [x] Core data structures
- [x] JSON loading and parsing
- [x] IoU calculation
- [x] Precision, Recall, F1 metrics
- [x] Confidence thresholding
- [ ] Full mAP/AP50/AP75 implementation
- [ ] Multi-class evaluation
- [ ] Keypoint detection support
- [ ] Instance segmentation support
- [ ] Performance benchmarks
- [ ] Visualization utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [COCO Dataset](https://cocodataset.org/)
- [COCO Evaluation](https://cocodataset.org/#detection-eval)
- [pycocotools](https://github.com/cocodataset/cocoapi)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{coco_eval_rs,
  title = {coco-eval: Rust library for COCO evaluation metrics},
  author = {George Pearse},
  year = {2025},
  url = {https://github.com/GeorgePearse/coco-eval}
}
```
