# Performance

`coco-eval` is designed for high performance, targeting 10x+ speedup over the Python reference implementation.

## Performance Goals

| Metric | Python | Rust Target | Status |
|--------|--------|-------------|--------|
| Evaluation (1000 images) | ~1.0s | <0.1s | In Progress |
| Threshold optimization | ~10s | <1s | In Progress |
| Memory usage | Baseline | 50% of Python | In Progress |

## Optimization Strategies

### Zero-Copy Operations

Minimize allocations by borrowing instead of cloning:

```rust
// Good - zero-copy
fn process(data: &[f64]) -> f64 {
    data.iter().sum()
}

// Avoid - unnecessary copy
fn process(data: Vec<f64>) -> f64 {
    data.iter().sum()
}
```

### Parallel Processing

Use Rayon for data parallelism:

```rust
use rayon::prelude::*;

let results: Vec<_> = images
    .par_iter()
    .map(|image| evaluate(image))
    .collect();
```

### Vectorization

Leverage SIMD when possible:

```rust
use ndarray::Array;

// ndarray uses SIMD for element-wise operations
let result = &array1 * &array2;
```

### Memory Layout

Use struct-of-arrays for better cache locality:

```rust
// Array-of-structs (less cache-friendly)
struct Prediction {
    x: f64,
    y: f64,
    confidence: f64,
}
let predictions: Vec<Prediction> = ...;

// Struct-of-arrays (better cache locality)
struct Predictions {
    x: Vec<f64>,
    y: Vec<f64>,
    confidence: Vec<f64>,
}
```

## Benchmarking

### Run Benchmarks

```bash
cargo bench
```

### Benchmark Specific Functions

```bash
cargo bench --bench my_benchmark
```

### Profiling

Use `flamegraph` for profiling:

```bash
cargo install flamegraph
cargo flamegraph --bench my_benchmark
```

## Performance Testing

### Comparison with Python

Test setup for comparing against Python reference:

1. Generate test dataset
2. Run Python evaluation and record time
3. Run Rust evaluation and record time
4. Verify results match (within tolerance)
5. Calculate speedup

Example results (target):

```
Dataset: 1000 images, 5000 predictions, 10 classes
Python: 1.23s
Rust:   0.09s
Speedup: 13.7x ✓
```

## Profiling Tips

### Identify Hotspots

```bash
cargo build --release
perf record --call-graph dwarf target/release/benchmark
perf report
```

### Memory Profiling

```bash
cargo install heaptrack
heaptrack target/release/benchmark
heaptrack --analyze heaptrack.benchmark.*.gz
```

## Optimization Checklist

When optimizing code:

- [ ] Profile first - don't guess
- [ ] Focus on hot paths
- [ ] Benchmark before and after
- [ ] Verify correctness isn't compromised
- [ ] Document optimization techniques used

## Known Bottlenecks

Areas that need optimization (to be identified as implementation progresses):

- IoU computation for large numbers of boxes
- Threshold grid search
- DataFrame operations
- Memory allocations

## Future Optimizations

Potential improvements:

- SIMD for IoU calculations
- GPU acceleration for large-scale evaluation
- Incremental evaluation
- Cached intermediate results

## Resources

- [The Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Rayon Documentation](https://docs.rs/rayon/)
- [ndarray Performance Guide](https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html)
