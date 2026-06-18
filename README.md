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
- **Automatic parallelism** via Rayon for large matrices

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

| Expression    | Returns                      | Description                         |
|---------------|------------------------------|-------------------------------------|
| `&a + &b`     | `Matrix`                     | Matrix addition (panics on error)   |
| `a.try_add(&b)`| `Result<Matrix, MatrixError>`| Safe matrix addition                |
| `&a - &b`     | `Matrix`                     | Matrix subtraction (panics on error)|
| `a.try_sub(&b)`| `Result<Matrix, MatrixError>`| Safe matrix subtraction             |
| `&a * &b`     | `Matrix`                     | Matrix multiplication (panics)      |
| `a.try_mul(&b)`| `Result<Matrix, MatrixError>`| Safe matrix multiplication          |
| `&a * 2.0`    | `Matrix`                     | Scalar multiplication               |
| `2.0 * &a`    | `Matrix`                     | Scalar multiplication (commutative) |

All operators are also available in owned form (`a + b`, `a * b`, etc.)
so you can choose based on whether you need to reuse the original.

### Indexing

```rust
let val = matrix[(i, j)];      // read element
matrix[(i, j)] = 3.14;         // write element
```

### Mathematical Operations

| Method           | Returns                       | Description                              |
|------------------|-------------------------------|------------------------------------------|
| `.transpose()`   | `Matrix`                      | Swap rows and columns                    |
| `.hadamard(&b)`  | `Result<Matrix, MatrixError>` | Element-wise multiplication              |
| `.norm()`        | `f64`                         | Frobenius norm                           |
| `.det()`         | `Result<f64, MatrixError>`    | Determinant (square matrices only)       |
| `.inverse()`     | `Result<Matrix, MatrixError>` | Inverse via Gauss-Jordan                 |

---

## Error Handling

NumRS uses a custom `MatrixError` enum for robust error handling. While standard operators (`+`, `-`, `*`) will panic with a descriptive message on invalid input (like shape mismatches), the library provides `try_*` variants and `Result`-returning methods for safe contexts.

```rust
use numrs::{ns_array, MatrixError};

let a = ns_array![[1, 2]];
let b = ns_array![[1, 2, 3]];

match a.try_add(&b) {
    Err(MatrixError::ShapeMismatch { expected, found }) => {
        println!("Error: expected {:?}, got {:?}", expected, found);
    }
    _ => unreachable!()
}
```

---

## Parallel Execution

NumRS automatically parallelizes operations on large matrices using [Rayon](https://github.com/rayon-rs/rayon).

- **Threshold**: Operations switch to parallel mode when the number of elements reaches `PAR_THRESHOLD` (currently 1024).
- **Tuning**: You can find this constant in `src/parallel.rs`.
- **Operations**: Addition, subtraction, multiplication, transposition, Hadamard product, and Frobenius norm are all parallelized.

---

## Project Structure

```
src/
├── main.rs          # Entry point and demos
├── lib.rs           # Crate root and exports
├── matrix.rs        # Matrix struct, Display, ns_array! macro, constructors
├── math.rs          # Mathematical operations
├── error.rs         # MatrixError enum and Error implementation
├── parallel.rs      # Parallelization utilities and thresholds
└── ops/
    ├── mod.rs
    ├── add.rs       # Add trait and try_add
    ├── sub.rs       # Sub trait and try_sub
    ├── mul.rs       # Mul trait (matrix × matrix) and try_mul
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
- [x] Automatic parallelism via Rayon
- [x] Comprehensive error handling with `MatrixError`
- [ ] Eigenvalues and eigenvectors
- [ ] LU / QR decomposition
- [ ] Tensor support
- [ ] ML layer: ReLU, softmax, sigmoid, gradient tracking

---

## License

MIT
