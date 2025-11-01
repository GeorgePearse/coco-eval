# Quick Start

This guide will help you get started with `coco-eval` in minutes.

## Basic Evaluation

Here's a complete example of evaluating object detection predictions:

```rust
use coco_eval::{Evaluator, Prediction, GroundTruth, BoundingBox};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define ground truth annotations
    let ground_truth = vec![
        GroundTruth {
            image_id: "img_001".to_string(),
            bbox: BoundingBox::new(10.0, 10.0, 50.0, 50.0),
            class_id: 1,
        },
        GroundTruth {
            image_id: "img_001".to_string(),
            bbox: BoundingBox::new(100.0, 100.0, 150.0, 150.0),
            class_id: 2,
        },
    ];

    // Define predictions
    let predictions = vec![
        Prediction {
            image_id: "img_001".to_string(),
            bbox: BoundingBox::new(12.0, 11.0, 52.0, 51.0),
            class_id: 1,
            confidence: 0.95,
        },
        Prediction {
            image_id: "img_001".to_string(),
            bbox: BoundingBox::new(105.0, 105.0, 155.0, 155.0),
            class_id: 2,
            confidence: 0.87,
        },
    ];

    // Create evaluator
    let evaluator = Evaluator::new()
        .iou_threshold(0.5)
        .confidence_threshold(0.5)
        .build();

    // Evaluate
    let results = evaluator.evaluate(&predictions, &ground_truth)?;

    // Print results
    println!("Precision: {:.3}", results.precision);
    println!("Recall: {:.3}", results.recall);
    println!("F1 Score: {:.3}", results.f1_score);
    println!("True Positives: {}", results.true_positives);
    println!("False Positives: {}", results.false_positives);
    println!("False Negatives: {}", results.false_negatives);

    Ok(())
}
```

## Loading from Files

Load predictions and ground truth from JSON files:

```rust
use coco_eval::{Evaluator, load_predictions, load_ground_truth};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load from JSON files
    let predictions = load_predictions("predictions.json")?;
    let ground_truth = load_ground_truth("ground_truth.json")?;

    // Evaluate
    let evaluator = Evaluator::default();
    let results = evaluator.evaluate(&predictions, &ground_truth)?;

    Ok(())
}
```

## Threshold Optimization

Find optimal confidence thresholds for each class:

```rust
use coco_eval::Evaluator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let predictions = load_predictions("predictions.json")?;
    let ground_truth = load_ground_truth("ground_truth.json")?;

    // Grid search over confidence thresholds
    let thresholds = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    let evaluator = Evaluator::new()
        .iou_threshold(0.5)
        .confidence_thresholds(thresholds)
        .build();

    let optimized = evaluator.optimize_thresholds(&predictions, &ground_truth)?;

    // Print optimal thresholds per class
    for (class_id, threshold) in optimized.optimal_thresholds {
        println!("Class {}: optimal threshold = {:.3}", class_id, threshold);
    }

    Ok(())
}
```

## Per-Class Metrics

Get detailed metrics for each class:

```rust
use coco_eval::Evaluator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let predictions = load_predictions("predictions.json")?;
    let ground_truth = load_ground_truth("ground_truth.json")?;

    let evaluator = Evaluator::default();
    let results = evaluator.evaluate(&predictions, &ground_truth)?;

    // Print per-class metrics
    for class_metrics in results.per_class {
        println!("Class {}: P={:.3}, R={:.3}, F1={:.3}",
            class_metrics.class_id,
            class_metrics.precision,
            class_metrics.recall,
            class_metrics.f1_score
        );
    }

    Ok(())
}
```

## Next Steps

- Explore the [API Reference](../guide/api.md) for detailed documentation
- Check out more [Examples](../guide/examples.md)
- Learn about [Performance Optimization](../development/performance.md)
