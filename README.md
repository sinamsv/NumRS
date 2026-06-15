# NumRS 🦀

A lightweight linear algebra library built from scratch in Rust —
with an intuitive macro syntax, colored terminal output, and clean modular design.

---

## Features

- `ns_array!` macro — write matrices like Python
- Colored terminal display with per-column alignment
- Full operator support for both owned and reference types (`+` `-` `*`)
- Scalar multiplication in both directions (`matrix * 2.0` and `2.0 * matrix`)
- Utility constructors: `zeros`, `ones`, `eye`, `fill`
- Direct element access via `matrix[(i, j)]`
- Mathematical operations: transpose, Hadamard, norm, determinant, inverse

---

## Quick Start

```bash
git clone https://github.com/sinamsv/NumRS.git
```

```bash
cargo run
```

---

## API Reference

### Constructors

| Method                    | Description                        |
|---------------------------|------------------------------------|
| `ns_array![[...], [...]]` | Create matrix from literal values  |
| `Matrix::zeros(r, c)`     | Matrix of zeros                    |
| `Matrix::ones(r, c)`      | Matrix of ones                     |
| `Matrix::eye(n)`          | n×n identity matrix                |
| `Matrix::fill(r, c, v)`   | Matrix filled with a specific value|

### Operators

| Expression    | Description                         |
|---------------|-------------------------------------|
| `&a + &b`     | Matrix addition                     |
| `&a - &b`     | Matrix subtraction                  |
| `&a * &b`     | Matrix multiplication (dot product) |
| `&a * 2.0`    | Scalar multiplication               |
| `2.0 * &a`    | Scalar multiplication (commutative) |

All operators are also available in owned form (`a + b`, `a * b`, etc.)
so you can choose based on whether you need to reuse the original.

### Indexing

```rust
let val = matrix[(i, j)];      // read element
matrix[(i, j)] = 3.14;         // write element
```

### Mathematical Operations

| Method           | Returns         | Description                              |
|------------------|-----------------|------------------------------------------|
| `.transpose()`   | `Matrix`        | Swap rows and columns                    |
| `.hadamard(&b)`  | `Matrix`        | Element-wise multiplication              |
| `.norm()`        | `f64`           | Frobenius norm                           |
| `.det()`         | `f64`           | Determinant (square matrices only)       |
| `.inverse()`     | `Option<Matrix>`| Inverse via Gauss-Jordan; `None` if singular |

---

## Project Structure

```
src/
├── main.rs          # Entry point and demos
├── matrix.rs        # Matrix struct, Display, ns_array! macro, constructors
├── math.rs          # Mathematical operations
└── ops/
    ├── mod.rs
    ├── add.rs       # Add trait
    ├── sub.rs       # Sub trait
    ├── mul.rs       # Mul trait (matrix × matrix)
    ├── scalar.rs    # Mul<f64> and f64 * Matrix
    └── index.rs     # Index and IndexMut traits
```

---

## What You Can Build With This

| Use Case                  | Operations needed                  |
|---------------------------|------------------------------------|
| Solve linear systems Ax=b | `inverse()` + `*`                  |
| Hill Cipher (cryptography)| `*`, `det()`, `inverse()`          |
| 2D/3D transformations     | `*`, constructors                  |
| Markov chain simulation   | `*` repeatedly on state vector     |
| Least squares fitting     | `transpose()`, `inverse()`, `*`    |
| Graph path counting       | Matrix powers via `*`              |

---

## Roadmap

- [x] Matrix struct with colored Display
- [x] Add, Sub, Mul operators (owned + reference)
- [x] Scalar multiplication
- [x] Constructors: zeros, ones, eye, fill
- [x] Index / IndexMut
- [x] Transpose, Hadamard, Norm, Determinant, Inverse
- [ ] Eigenvalues and eigenvectors
- [ ] LU / QR decomposition
- [ ] Tensor support
- [ ] ML layer: ReLU, softmax, sigmoid, gradient tracking

---

## License

MIT
