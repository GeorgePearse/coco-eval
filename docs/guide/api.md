# API Reference

!!! note
    This is a placeholder page. The Rust implementation is in progress.

    For the Python reference implementation API, see `references/docs/README.md`

## Evaluator

Main evaluation engine for computing object detection metrics.

Coming soon...

## Data Structures

### BoundingBox

```rust
pub struct BoundingBox {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}
```

### Prediction

```rust
pub struct Prediction {
    pub image_id: String,
    pub bbox: BoundingBox,
    pub class_id: i64,
    pub confidence: f64,
}
```

### GroundTruth

```rust
pub struct GroundTruth {
    pub image_id: String,
    pub bbox: BoundingBox,
    pub class_id: i64,
}
```

## Results

### EvaluationResult

```rust
pub struct EvaluationResult {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub per_class: Vec<ClassMetrics>,
}
```

More details coming soon as the Rust implementation progresses.
