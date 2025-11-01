# Reference Implementation - READ ONLY

⚠️ **DO NOT MODIFY ANY FILES IN THIS DIRECTORY** ⚠️

## Purpose

This directory contains the original Python implementation of the evaluation metrics from the BinIt `core` repository. It serves as:

1. **Specification**: The authoritative reference for expected behavior
2. **Validation**: Test cases to ensure the Rust port is correct
3. **Performance Baseline**: For benchmarking Rust improvements
4. **API Reference**: Design guide for the Rust API

## Structure

```
references/
├── python/              # Python source code (8 files)
│   ├── eval_metrics.py      # Core implementation (1034 lines)
│   ├── __init__.py          # Package exports
│   ├── conversions.py       # Data conversion utilities
│   ├── db.py                # Database utilities (BinIt-specific)
│   └── test_*.py            # Comprehensive test suite
├── docs/                # Documentation and configuration
│   ├── README.md            # Python API documentation
│   └── pyproject.toml       # Dependency reference
└── examples/            # Real-world usage examples
    ├── offline_inference_metrics.py
    └── ml_deployment_infer_on_val.py
```

## How to Use This Reference

### ✅ DO:

- **Read** the Python code to understand algorithms and edge cases
- **Study** the test cases to understand expected behavior
- **Review** the examples to see real-world usage patterns
- **Reference** `pyproject.toml` to understand Python dependencies
- **Compare** Rust output against Python output for validation

### ❌ DO NOT:

- **Modify** any files in this directory
- **Copy-paste** code directly (port it properly to Rust instead)
- **Execute** this code as part of the Rust build process
- **Update** dependencies or fix bugs here

## If You Find Issues

If you discover bugs or issues in the reference implementation:

1. **Document** the issue in a GitHub issue
2. **Fix it upstream** in the BinIt `core` repository at:
   - Path: `/core/lib/python/evaluation/`
   - Repository: Internal BinIt core monorepo
3. **DO NOT** fix it here
4. Optionally: Work around it in the Rust implementation and document the difference

## Syncing with Upstream

If the Python implementation is updated in the `core` repository:

1. Copy the updated files from `core/lib/python/evaluation/` to this directory
2. Note the changes in a commit message
3. Update Rust implementation if needed to maintain compatibility
4. Run tests to ensure Rust port still validates correctly

## Key Python Dependencies

From `docs/pyproject.toml`, the Python version uses:

- `numpy==2.2.6` → Rust equivalent: `ndarray`
- `polars==1.2.0` → Rust equivalent: `polars`
- `matplotlib==3.9.2` → Rust equivalent: `plotters` or `plotly` (optional)
- `scikit-learn==1.7.1` → Rust equivalent: Custom implementation
- `supervision==0.26.1` → Rust equivalent: Custom NMS implementation
- `loguru==0.7.3` → Rust equivalent: `log` + `env_logger`
- `psycopg[binary,pool]==3.2.10` → Optional (database integration is BinIt-specific)

## Core Features to Port

From `python/eval_metrics.py`:

1. **Evaluator Class**: Main evaluation engine
2. **IoU Computation**: Intersection over Union calculation
3. **NMS Application**: Non-maximum suppression
4. **Threshold Optimization**: Grid search for optimal confidence thresholds
5. **F-beta Score**: Precision/recall trade-off calculation
6. **Confusion Matrix**: Visualization and statistics
7. **Camera-wise Stats**: Performance breakdown by camera/source

## Testing

The test files (`test_*.py`) contain:

- **test_eval_metrics.py**: Core evaluator tests
- **test_eval_metrics_additional.py**: Additional edge cases
- **test_conversions.py**: Data conversion tests
- **test_db.py**: Database utility tests (BinIt-specific)

Use these tests to validate that the Rust implementation produces the same results.

---

**Remember: This directory is READ-ONLY reference material. All development happens in the Rust codebase.**
