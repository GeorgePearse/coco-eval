# Python Reference Implementation

!!! warning "Read-Only Reference"
    The Python implementation in the `references/` directory is **READ-ONLY**. Never modify these files. They serve as the specification for the Rust port.

## Overview

The `references/` directory contains the original Python implementation from the BinIt `core` repository. This production-tested code serves multiple purposes:

1. **Specification** - Defines the expected behavior for the Rust port
2. **Test Validation** - Provides test cases to ensure correctness
3. **Performance Baseline** - Establishes benchmarks for improvement
4. **API Reference** - Guides the design of the Rust API

## Directory Structure

```
references/
├── python/              # Python source code
│   ├── eval_metrics.py      # Core implementation (1034 lines)
│   ├── __init__.py          # Package exports
│   ├── conversions.py       # Data conversion utilities
│   ├── db.py                # Database utilities (BinIt-specific)
│   └── test_*.py            # Comprehensive test suite
├── docs/                # Documentation
│   ├── README.md            # Python API documentation
│   └── pyproject.toml       # Dependency reference
└── examples/            # Real-world usage
    ├── offline_inference_metrics.py
    └── ml_deployment_infer_on_val.py
```

## Core Components

### Evaluator Class

The main `Evaluator` class in `python/eval_metrics.py` provides:

- IoU-based matching between predictions and ground truth
- Configurable IoU and confidence thresholds
- Non-maximum suppression (NMS) application
- F-beta score calculation
- Confusion matrix generation
- Camera-wise statistics

### Key Methods

```python
class Evaluator:
    def __init__(
        self,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        nms_threshold: Optional[float] = None,
    ):
        ...

    def evaluate(
        self,
        predictions_df: pl.DataFrame,
        ground_truth_df: pl.DataFrame,
    ) -> Dict[str, Any]:
        ...

    def optimize_thresholds(
        self,
        predictions_df: pl.DataFrame,
        ground_truth_df: pl.DataFrame,
        thresholds: List[float],
    ) -> Dict[int, float]:
        ...
```

### Data Format

The Python implementation uses Polars DataFrames with the following schema:

**Predictions DataFrame:**
```python
{
    "image_id": pl.Utf8,
    "bbox_x1": pl.Float64,
    "bbox_y1": pl.Float64,
    "bbox_x2": pl.Float64,
    "bbox_y2": pl.Float64,
    "class_id": pl.Int64,
    "confidence": pl.Float64,
}
```

**Ground Truth DataFrame:**
```python
{
    "image_id": pl.Utf8,
    "bbox_x1": pl.Float64,
    "bbox_y1": pl.Float64,
    "bbox_x2": pl.Float64,
    "bbox_y2": pl.Float64,
    "class_id": pl.Int64,
}
```

## Python Dependencies

From `docs/pyproject.toml`:

| Python Package | Version | Rust Equivalent |
|---------------|---------|-----------------|
| numpy | 2.2.6 | `ndarray` |
| polars | 1.2.0 | `polars` |
| matplotlib | 3.9.2 | `plotters` or `plotly` |
| scikit-learn | 1.7.1 | Custom implementation |
| supervision | 0.26.1 | Custom NMS |
| loguru | 0.7.3 | `log` + `env_logger` |
| psycopg | 3.2.10 | Optional (DB-specific) |

## Test Cases

The test files provide comprehensive coverage:

- `test_eval_metrics.py` - Core evaluator tests
- `test_eval_metrics_additional.py` - Additional edge cases
- `test_conversions.py` - Data conversion tests
- `test_db.py` - Database utility tests

These tests should be used to validate the Rust implementation produces identical results (within floating-point tolerance).

## Real-World Usage

The `examples/` directory shows how the evaluation package is used in production:

### Offline Inference Metrics

From `offline_inference_metrics.py` - Shows integration with BinIt's offline inference pipeline:

- Loading predictions from ClickHouse
- Converting inference results to evaluation format
- Computing metrics per camera
- Storing results back to PostgreSQL

### ML Deployment Validation

From `ml_deployment_infer_on_val.py` - Shows ML model validation workflow:

- Running inference on validation dataset
- Evaluating model performance
- Threshold optimization
- Generating confusion matrices

## Porting Guidelines

When porting features from Python to Rust:

1. **Read the Python code** to understand the algorithm
2. **Review associated tests** to understand edge cases
3. **Implement in Rust** with idiomatic patterns
4. **Validate against Python tests** - results should match within `1e-6` relative error
5. **Optimize for performance** - leverage Rust's strengths
6. **Document differences** - note any intentional deviations

## Performance Expectations

The Python implementation is already optimized:

- Uses Polars (faster than Pandas)
- Vectorized NumPy operations
- Efficient IoU computation

The Rust port should achieve **10x+ speedup** through:

- Zero-copy operations
- Parallel processing with Rayon
- Memory-efficient data structures
- Compile-time optimizations

## Questions?

If you find issues in the Python reference:

1. **Do not modify** the reference implementation
2. **Document** the issue in a GitHub issue
3. **Fix upstream** in the BinIt core repository
4. Optionally **work around** in Rust and document the difference

See the [Porting Guide](porting.md) for detailed instructions on translating Python code to Rust.
