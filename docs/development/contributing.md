# Contributing

Thank you for your interest in contributing to `coco-eval`!

## Development Setup

1. **Clone the repository:**

```bash
git clone https://github.com/GeorgePearse/coco-eval.git
cd coco-eval
```

2. **Install Rust:**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

3. **Build the project:**

```bash
cargo build
```

4. **Run tests:**

```bash
cargo test
```

## Code Standards

### Formatting

Use `rustfmt` for consistent formatting:

```bash
cargo fmt
```

### Linting

Run `clippy` to catch common issues:

```bash
cargo clippy -- -D warnings
```

### Documentation

Add doc comments to all public APIs:

```rust
/// Brief description
///
/// # Arguments
///
/// * `arg` - Argument description
///
/// # Returns
///
/// Return value description
///
/// # Example
///
/// ```
/// use coco_eval::function;
/// let result = function(42);
/// ```
pub fn function(arg: i32) -> i32 {
    ...
}
```

## Testing

### Unit Tests

Add tests for all new functionality:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        // Test implementation
    }
}
```

### Integration Tests

Add integration tests in `tests/`:

```rust
// tests/integration_test.rs
use coco_eval::*;

#[test]
fn test_end_to_end() {
    // Integration test
}
```

### Benchmarks

Add benchmarks for performance-critical code:

```rust
// benches/benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use coco_eval::*;

fn benchmark_function(c: &mut Criterion) {
    c.bench_function("my_function", |b| {
        b.iter(|| my_function(black_box(42)))
    });
}

criterion_group!(benches, benchmark_function);
criterion_main!(benches);
```

## Validation Against Python

New features should match the Python reference implementation:

1. Identify equivalent Python functionality in `references/python/`
2. Review Python test cases
3. Implement in Rust
4. Verify results match (within `1e-6` relative error)
5. Document any intentional differences

## Pull Request Process

1. **Create a branch:**

```bash
git checkout -b feature/my-feature
```

2. **Make your changes**

3. **Run all checks:**

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo bench
```

4. **Commit with conventional commits:**

```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug"
git commit -m "docs: update documentation"
```

5. **Push and create PR:**

```bash
git push origin feature/my-feature
```

Then create a pull request on GitHub.

## PR Guidelines

- Keep PRs focused and small
- Include tests for new functionality
- Update documentation as needed
- Ensure all CI checks pass
- Reference related issues

## Protected Files

!!! danger "Do Not Modify"
    **Never modify files in the `references/` directory.** This is the read-only Python reference implementation.

    If you find issues in the reference, document them in a GitHub issue and fix them in the upstream BinIt core repository.

## Getting Help

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Ask questions in issue discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
