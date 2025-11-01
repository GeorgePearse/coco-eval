# Installation

## Prerequisites

- Rust 1.70 or later
- Cargo (comes with Rust)

## Install Rust

If you don't have Rust installed, get it from [rustup.rs](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Add coco-eval to Your Project

Add `coco-eval` to your `Cargo.toml`:

```toml
[dependencies]
coco-eval = "0.1"
```

## Development Setup

To contribute to `coco-eval`, clone the repository:

```bash
git clone https://github.com/GeorgePearse/coco-eval.git
cd coco-eval
```

### Build the Project

```bash
cargo build
```

### Run Tests

```bash
cargo test
```

### Run Benchmarks

```bash
cargo bench
```

### Generate Documentation

```bash
cargo doc --open
```

## Optional Dependencies

### Visualization Features

For confusion matrix visualization (requires additional dependencies):

```toml
[dependencies]
coco-eval = { version = "0.1", features = ["visualization"] }
```

### Serialization Support

For additional serialization formats:

```toml
[dependencies]
coco-eval = { version = "0.1", features = ["serde"] }
```

## Verify Installation

Create a simple test program:

```rust
use coco_eval::Evaluator;

fn main() {
    println!("coco-eval is installed!");
    let evaluator = Evaluator::default();
    println!("Evaluator created successfully");
}
```

Run it:

```bash
cargo run
```

If you see the success messages, you're ready to go!

## Next Steps

- [Quick Start Tutorial](quickstart.md)
- [API Reference](../guide/api.md)
- [Examples](../guide/examples.md)
