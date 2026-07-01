# NumRS 🦀

A lightweight, high-performance linear algebra and tensor library built from scratch in Rust. NumRS provides an intuitive API, automatic parallelism for large matrices, and a clean modular design.

---

## Key Features

- **2D Matrix & N-D Tensor**: Specialized `Matrix` for linear algebra and generic `Tensor<T>` for multi-dimensional data.
- **Intuitive Syntax**: Create matrices with the `ns_array!` macro — just like Python/NumPy.
- **Automatic Parallelism**: Silently scales across CPU cores using Rayon for large-scale computations.
- **Robust Math**: Transpose, Determinant, Inverse (Gauss-Jordan), Norm, and Hadamard product.
- **Beautiful Output**: Colored terminal display with automatic column alignment.
- **Reference-First Operators**: Clean arithmetic (`a + b`, `a * b`) with zero unnecessary clones.

---

## Quick Start

```rust
use numrs::{ns_array, Matrix};

fn main() {
    let a = ns_array![[1, 2], [3, 4]];
    let b = Matrix::eye(2);

    let result = &a * &b;
    println!("A x I =\n{}", result);
}
```

---

## Documentation & Learning

- **[Usage Guide](USAGE.md)** — Detailed API documentation for all modules.
- **[Examples](EXAMPLES.md)** — Practical recipes for solving linear systems, Markov chains, transformations, and more.
- **[Changelog](CHANGELOG.md)** — History of releases and recent changes.

---

## Roadmap

- [x] Matrix struct with colored Display
- [x] Full operator support (Add, Sub, Mul, Scalar)
- [x] Constructors (zeros, ones, eye, rand)
- [x] Determinant and Inverse
- [x] Automatic parallelism via Rayon
- [x] Comprehensive error handling
- [x] **Tensor Support** — N-D generic `Tensor<T>` with full error handling (`TensorError`, `try_*` methods)
- [ ] Eigenvalues and eigenvectors
- [ ] LU / QR decomposition
- [ ] ML layer: ReLU, softmax, gradient tracking

---

## License

MIT
