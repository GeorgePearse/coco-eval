# Testing

Comprehensive testing strategy for `coco-eval`.

## Test Types

### Unit Tests

Test individual functions and methods:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_area() {
        let bbox = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(bbox.area(), 100.0);
    }

    #[test]
    fn test_iou_identical_boxes() {
        let box1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let box2 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let iou = calculate_iou(&box1, &box2);
        assert!((iou - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_iou_no_overlap() {
        let box1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let box2 = BoundingBox::new(20.0, 20.0, 30.0, 30.0);
        let iou = calculate_iou(&box1, &box2);
        assert_eq!(iou, 0.0);
    }
}
```

### Integration Tests

Test complete workflows in `tests/`:

```rust
// tests/integration_test.rs
use coco_eval::*;

#[test]
fn test_end_to_end_evaluation() {
    let predictions = vec![
        Prediction {
            image_id: "img_001".to_string(),
            bbox: BoundingBox::new(10.0, 10.0, 50.0, 50.0),
            class_id: 1,
            confidence: 0.95,
        },
    ];

    let ground_truth = vec![
        GroundTruth {
            image_id: "img_001".to_string(),
            bbox: BoundingBox::new(12.0, 12.0, 52.0, 52.0),
            class_id: 1,
        },
    ];

    let evaluator = Evaluator::default();
    let results = evaluator.evaluate(&predictions, &ground_truth).unwrap();

    assert!(results.precision > 0.9);
    assert!(results.recall > 0.9);
}
```

### Property-Based Testing

Use `proptest` for edge cases:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_iou_commutative(
        x1 in 0.0..100.0,
        y1 in 0.0..100.0,
        w1 in 1.0..50.0,
        h1 in 1.0..50.0,
        x2 in 0.0..100.0,
        y2 in 0.0..100.0,
        w2 in 1.0..50.0,
        h2 in 1.0..50.0,
    ) {
        let box1 = BoundingBox::new(x1, y1, x1 + w1, y1 + h1);
        let box2 = BoundingBox::new(x2, y2, x2 + w2, y2 + h2);

        let iou1 = calculate_iou(&box1, &box2);
        let iou2 = calculate_iou(&box2, &box1);

        assert!((iou1 - iou2).abs() < 1e-10);
    }

    #[test]
    fn test_iou_bounds(
        x1 in 0.0..100.0,
        y1 in 0.0..100.0,
        w1 in 1.0..50.0,
        h1 in 1.0..50.0,
        x2 in 0.0..100.0,
        y2 in 0.0..100.0,
        w2 in 1.0..50.0,
        h2 in 1.0..50.0,
    ) {
        let box1 = BoundingBox::new(x1, y1, x1 + w1, y1 + h1);
        let box2 = BoundingBox::new(x2, y2, x2 + w2, y2 + h2);

        let iou = calculate_iou(&box1, &box2);

        assert!(iou >= 0.0);
        assert!(iou <= 1.0);
    }
}
```

## Validation Against Python

Critical: Results must match the Python reference implementation.

### Tolerance

Floating-point comparisons should use relative tolerance:

```rust
fn assert_close(actual: f64, expected: f64, rel_tol: f64) {
    let diff = (actual - expected).abs();
    let max_val = actual.abs().max(expected.abs());
    assert!(
        diff <= rel_tol * max_val,
        "Values not close: {} vs {} (diff: {})",
        actual, expected, diff
    );
}

#[test]
fn test_matches_python() {
    let result = compute_metric();
    let python_result = 0.857142;  // From Python reference
    assert_close(result, python_result, 1e-6);
}
```

### Test Data

Use the same test data as Python:

```rust
// Load test data from JSON (same format as Python tests)
let test_data: TestCase = serde_json::from_str(include_str!("test_data.json"))?;

let result = evaluate(&test_data.predictions, &test_data.ground_truth)?;

// Compare with Python results
assert_close(result.precision, test_data.expected.precision, 1e-6);
assert_close(result.recall, test_data.expected.recall, 1e-6);
```

## Running Tests

### All Tests

```bash
cargo test
```

### Specific Test

```bash
cargo test test_iou_calculation
```

### With Output

```bash
cargo test -- --nocapture
```

### Release Mode

```bash
cargo test --release
```

## Test Coverage

### Install tarpaulin

```bash
cargo install cargo-tarpaulin
```

### Generate Coverage Report

```bash
cargo tarpaulin --out Html
```

### Coverage Goals

- **Core functionality**: 90%+ coverage
- **Edge cases**: Well-tested with property tests
- **Integration**: Complete workflows tested

## Continuous Integration

GitHub Actions runs:

- `cargo test` - All tests
- `cargo clippy` - Linting
- `cargo fmt --check` - Formatting
- `cargo tarpaulin` - Coverage

## Test Organization

```
coco-eval/
├── src/
│   ├── lib.rs              # Unit tests inline
│   ├── evaluator.rs        # Unit tests inline
│   └── ...
├── tests/
│   ├── integration_test.rs
│   ├── python_compat_test.rs
│   └── test_data/
│       └── *.json
└── benches/
    └── benchmark.rs
```

## Writing Good Tests

### Test Names

Use descriptive names:

```rust
#[test]
fn test_iou_identical_boxes_returns_one() { ... }

#[test]
fn test_iou_no_overlap_returns_zero() { ... }

#[test]
fn test_evaluator_rejects_invalid_iou_threshold() { ... }
```

### Test Organization

Group related tests:

```rust
mod bounding_box_tests {
    use super::*;

    mod area {
        use super::*;

        #[test]
        fn test_positive_area() { ... }

        #[test]
        fn test_zero_area() { ... }
    }

    mod iou {
        use super::*;

        #[test]
        fn test_identical() { ... }

        #[test]
        fn test_no_overlap() { ... }

        #[test]
        fn test_partial_overlap() { ... }
    }
}
```

### Error Cases

Test error handling:

```rust
#[test]
#[should_panic(expected = "Invalid IoU threshold")]
fn test_invalid_iou_threshold() {
    Evaluator::new().iou_threshold(-0.5).build();
}

#[test]
fn test_empty_predictions() {
    let predictions = vec![];
    let ground_truth = vec![/* ... */];

    let evaluator = Evaluator::default();
    let result = evaluator.evaluate(&predictions, &ground_truth).unwrap();

    assert_eq!(result.precision, 0.0);
    assert_eq!(result.recall, 0.0);
}
```

## Resources

- [Rust Testing Guide](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [proptest Documentation](https://docs.rs/proptest/)
- [cargo-tarpaulin](https://github.com/xd009642/tarpaulin)
