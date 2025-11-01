# coco-eval Agent Guidelines

## Project Overview

This repository contains a high-performance Rust implementation of COCO-style object detection evaluation metrics with confidence threshold optimization. The goal is to port the Python implementation from the BinIt `core` repository to Rust for significantly improved performance.

## Critical Rules

### DO NOT TOUCH THE REFERENCE IMPLEMENTATION

**NEVER modify, edit, or update any files in the `references/` directory.**

The `references/` directory contains the original Python implementation from the `core` repository. This is **read-only reference material** that serves as:

- The specification for the Rust port
- Test case validation reference
- Performance baseline for benchmarking
- API design reference

**If you need to make changes to the evaluation logic:**
1. Make them in the Rust implementation under `src/`
2. Document differences from the Python version in comments
3. Update tests to validate the new behavior

**If you find issues in the Python reference:**
1. Document them in GitHub issues
2. Fix them in the upstream `core` repository
3. DO NOT modify the reference implementation here

### Repository Structure

```
coco-eval/
├── references/          # READ-ONLY: Python reference implementation
│   ├── python/         # Original Python source files
│   ├── docs/           # Python documentation and dependencies
│   └── examples/       # Real-world usage examples
├── src/                # Rust implementation (EDITABLE)
│   ├── lib.rs
│   ├── evaluator.rs
│   ├── metrics.rs
│   └── ...
├── tests/              # Rust tests (EDITABLE)
├── examples/           # Rust examples (EDITABLE)
├── benches/            # Performance benchmarks (EDITABLE)
└── Cargo.toml          # Rust dependencies (EDITABLE)
```

## Development Guidelines

### Rust Code Style

- **Follow Rust idioms**: Use `rustfmt` and `clippy` for all code
- **Zero-copy where possible**: Minimize allocations for performance
- **Type safety**: Leverage Rust's type system to prevent errors at compile time
- **Error handling**: Use `Result<T, E>` with custom error types via `thiserror`
- **Documentation**: All public APIs must have doc comments with examples

### Performance Requirements

This is a **performance-focused** Rust port. The implementation must be significantly faster than the Python version.

**Key optimizations:**
- Use `ndarray` for vectorized operations (equivalent to numpy)
- Leverage `polars` for DataFrame operations (already fast in Python, should be even faster in Rust)
- Implement parallel processing with `rayon` where applicable
- Minimize allocations in hot paths
- Profile and benchmark regularly using `criterion`

**Benchmarking:**
- Compare against Python reference implementation on identical datasets
- Document performance improvements in README
- Use `cargo bench` to track performance over time
- Aim for at least 10x speedup on typical workloads

### Testing Strategy

1. **Unit Tests**: Test individual functions and components
2. **Integration Tests**: Validate complete evaluation pipelines
3. **Compatibility Tests**: Ensure results match Python reference implementation
   - Use test cases from `references/python/test_*.py`
   - Parse or convert Python test data for Rust tests
   - Tolerate floating-point differences (relative error < 1e-6)

### API Design

- **Mirror Python API where sensible**: Make it easy for users familiar with the Python version
- **Improve where Rust allows**: Use builder patterns, iterators, and type safety
- **Provide multiple interfaces**:
  - High-level API for simple use cases
  - Low-level API for performance-critical scenarios
  - Serialization/deserialization for common formats (JSON, Parquet)

### Dependencies

**Core dependencies** (already in `Cargo.toml`):
- `serde` / `serde_json` - Serialization
- `ndarray` - Numerical arrays (numpy equivalent)
- `thiserror` - Error handling
- `log` - Logging

**Add as needed:**
- `polars` - DataFrame library (recommended for performance)
- `rayon` - Parallel iterators
- `plotters` or `plotly` - Visualization (if porting confusion matrix plotting)

### Validation Against Reference

When implementing features:

1. Read the Python implementation in `references/python/eval_metrics.py`
2. Understand the algorithm and edge cases
3. Implement in Rust with performance optimizations
4. Validate against Python test cases
5. Add Rust-specific tests for edge cases
6. Benchmark performance improvements

### Common Tasks

- **Format code**: `cargo fmt`
- **Lint code**: `cargo clippy -- -D warnings`
- **Run tests**: `cargo test`
- **Run benchmarks**: `cargo bench`
- **Build release**: `cargo build --release`
- **Generate docs**: `cargo doc --open`

### Documentation

- **README.md**: High-level overview, installation, quick start, performance comparisons
- **API docs**: Comprehensive doc comments with examples for all public APIs
- **Examples**: Working examples in `examples/` directory
- **CHANGELOG.md**: Track changes, especially API differences from Python version

## Reference Implementation Usage

**How to use the reference implementation:**

1. **Read the code**: Study algorithms, data structures, and edge cases
2. **Review tests**: Understand expected behavior and edge cases
3. **Check examples**: See how it's used in production systems
4. **Reference dependencies**: Understand what Python libraries do for Rust equivalents

**DO NOT:**
- Modify any files in `references/`
- Copy-paste Python code directly (port the logic properly to Rust)
- Import or execute the Python reference in the Rust build process

## Contributing

When making changes:

1. Ensure all tests pass: `cargo test`
2. Ensure code is formatted: `cargo fmt --check`
3. Ensure no clippy warnings: `cargo clippy -- -D warnings`
4. Add benchmarks for performance-critical code
5. Update documentation for API changes
6. Compare performance against Python baseline

## Questions or Issues

If you encounter ambiguity or need clarification:

1. Check the Python reference implementation
2. Review the upstream `core` repository documentation
3. Open a GitHub issue for discussion
4. Document your decision in code comments

---

**Remember: The `references/` directory is READ-ONLY. All development happens in `src/`, `tests/`, `examples/`, and `benches/`.**
