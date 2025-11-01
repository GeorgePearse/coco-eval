# Porting Guide

Guidelines for porting features from the Python reference implementation to Rust.

## General Process

1. **Study the Python code** - Understand the algorithm and edge cases
2. **Review tests** - Identify expected behavior and edge cases
3. **Design Rust API** - Create idiomatic Rust interfaces
4. **Implement** - Write the Rust code
5. **Validate** - Ensure results match Python (within tolerance)
6. **Optimize** - Leverage Rust performance features
7. **Document** - Add docs and examples

## Python to Rust Mapping

### Data Structures

| Python | Rust |
|--------|------|
| `pl.DataFrame` | `polars::DataFrame` |
| `np.ndarray` | `ndarray::Array` |
| `Dict[str, Any]` | `HashMap<String, Value>` or custom struct |
| `List[T]` | `Vec<T>` |
| `Optional[T]` | `Option<T>` |

### Functions

Python functions should map to Rust methods or functions:

```python
# Python
def calculate_iou(box1: Dict, box2: Dict) -> float:
    ...
```

```rust
// Rust
fn calculate_iou(box1: &BoundingBox, box2: &BoundingBox) -> f64 {
    ...
}
```

### Error Handling

Python exceptions should map to Rust Results:

```python
# Python
def evaluate(...):
    if not valid:
        raise ValueError("Invalid input")
    ...
```

```rust
// Rust
fn evaluate(...) -> Result<EvaluationResult, EvalError> {
    if !valid {
        return Err(EvalError::InvalidInput);
    }
    ...
}
```

## Performance Optimizations

### Use Zero-Copy Where Possible

Prefer borrowing over cloning:

```rust
// Good
fn process(data: &[f64]) -> f64 { ... }

// Avoid if possible
fn process(data: Vec<f64>) -> f64 { ... }
```

### Leverage Parallel Processing

Use Rayon for parallelizable operations:

```rust
use rayon::prelude::*;

let results: Vec<_> = data
    .par_iter()
    .map(|item| process(item))
    .collect();
```

### Prefer Iterators

Rust iterators are often faster than explicit loops:

```rust
// Good
let sum: f64 = values.iter().sum();

// Less efficient
let mut sum = 0.0;
for &value in &values {
    sum += value;
}
```

## Testing Strategy

### Unit Tests

Match Python test cases:

```rust
#[test]
fn test_iou_calculation() {
    let box1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
    let box2 = BoundingBox::new(5.0, 5.0, 15.0, 15.0);

    let iou = calculate_iou(&box1, &box2);

    // Match Python result within tolerance
    assert!((iou - 0.142857).abs() < 1e-6);
}
```

### Property-Based Testing

Use `proptest` for edge cases:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_iou_symmetric(
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
}
```

## Common Patterns

### Builder Pattern

Replace Python kwargs with builder:

```python
# Python
evaluator = Evaluator(
    iou_threshold=0.5,
    confidence_threshold=0.7,
    nms_threshold=0.3
)
```

```rust
// Rust
let evaluator = Evaluator::new()
    .iou_threshold(0.5)
    .confidence_threshold(0.7)
    .nms_threshold(0.3)
    .build();
```

### Type Safety

Use enums for fixed sets:

```python
# Python
def set_mode(mode: str):
    if mode not in ["precision", "recall", "f1"]:
        raise ValueError(...)
```

```rust
// Rust
enum OptimizationMode {
    Precision,
    Recall,
    F1,
}

fn set_mode(mode: OptimizationMode) {
    // Type system ensures valid mode
}
```

## Documentation

Add doc comments with examples:

```rust
/// Calculates the Intersection over Union (IoU) between two bounding boxes.
///
/// # Arguments
///
/// * `box1` - First bounding box
/// * `box2` - Second bounding box
///
/// # Returns
///
/// IoU value between 0.0 and 1.0
///
/// # Example
///
/// ```
/// use coco_eval::{BoundingBox, calculate_iou};
///
/// let box1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
/// let box2 = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
///
/// let iou = calculate_iou(&box1, &box2);
/// assert!((iou - 0.142857).abs() < 1e-6);
/// ```
pub fn calculate_iou(box1: &BoundingBox, box2: &BoundingBox) -> f64 {
    ...
}
```

## Resources

- [Python Reference](python.md)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
