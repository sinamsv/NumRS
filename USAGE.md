# NumRS — Usage Guide

This guide explains everything you need to know to use NumRS: what each
constructor, operator, and method does, how the two core types relate to
each other, and the internal design decisions.

---

## What is NumRS?

NumRS is a linear algebra and tensor library built from scratch in Rust,
designed for both ease of use and performance.

| Type        | Dimensions | Element type | What it's for                                   |
|-------------|------------|---------------|--------------------------------------------------|
| `Matrix`    | 2D only    | `f64` only    | Linear algebra: solving systems, transforms, crypto math |
| `Tensor<T>` | N-D, any rank | generic `T` | N-dimensional data — the basis for future AI/tensor work |

---

## Installation

NumRS can be included as a path or git dependency in your `Cargo.toml`:

```toml
[dependencies]
numrs = { path = "../NumRS" }
# or: numrs = { git = "https://github.com/sinamsv/NumRS.git" }
```

---

## Detailed API Documentation

### 1. `numrs::matrix` — 2D Matrix Operations

The `Matrix` type is the workhorse for 2D linear algebra. It uses a flat `Vec<f64>` in **row-major** order.

#### Constructors
- `ns_array![[1,2],[3,4]]`: Macro for intuitive matrix creation.
- `Matrix::new(rows, cols, data)`: Creates a matrix from flat data. Panics if `rows * cols != data.len()`.
- `Matrix::zeros(rows, cols)`, `Matrix::ones(rows, cols)`, `Matrix::fill(rows, cols, val)`: Utility constructors.
- `Matrix::eye(n)`: Creates an $n \times n$ identity matrix.
- `Matrix::rand(rows, cols)`, `Matrix::randn(rows, cols)`: Random matrices (Uniform and Normal).
- `Matrix::rand_seeded(rows, cols, seed)`, `Matrix::randn_seeded(rows, cols, seed)`: Deterministic random matrices.

#### Arithmetic & Operators
NumRS uses a **reference-first** operator model. All operators (`+`, `-`, `*`) work on:
- `&Matrix op &Matrix`
- `&Matrix op Matrix`
- `Matrix op &Matrix`
- `Matrix op Matrix`

**Note:** Standard operators panic on dimension mismatch with a descriptive `MatrixError` message. For non-panicking alternatives, see "Safe Variants".

#### Safe Variants
If you prefer `Result` over panics, use the `try_*` methods:
- `a.try_add(&b)` -> `Result<Matrix, MatrixError>`
- `a.try_sub(&b)` -> `Result<Matrix, MatrixError>`
- `a.try_mul(&b)` -> `Result<Matrix, MatrixError>`

### 2. `numrs::tensor` — N-Dimensional Tensors

`Tensor<T>` generalizes the row-major storage to N dimensions.

#### Usage
```rust
let t: Tensor<f32> = Tensor::zeros(&[2, 3, 4]); // A 3D tensor
let val = t[&[0, 1, 2]];                        // Stride-based indexing
```

#### Supported Operations
- **Basic Arithmetic**: `+`, `-` between tensors of the same shape.
- **Scalar**: `tensor * scalar` (multiplies every element).
- **Hadamard**: `a.hadamard(&b)` for element-wise multiplication.
- **Random**: `.rand()`, `.randn()`, and seeded variants similar to `Matrix`.

### 3. `numrs::math` — Advanced Mathematics

The `math` module provides complex operations for `Matrix`:
- `.transpose()`: Returns a new transposed matrix.
- `.det()`: Calculates the determinant using Gaussian elimination with partial pivoting.
- `.inverse()`: Calculates the Gauss-Jordan inverse. Returns `Result<Matrix, MatrixError>`.
- `.norm()`: Calculates the Frobenius norm.

### 4. `numrs::error` — Error Handling

NumRS uses the `MatrixError` enum to categorize failures:
- `ShapeMismatch`: Element-wise operation on incompatible shapes.
- `DimensionMismatch`: Matrix multiplication shape error.
- `NonSquare`: Operation requires a square matrix (e.g., `det`, `inverse`).
- `NonInvertible`: Matrix is singular.
- `IndexOutOfBounds`: Accessing an element outside the matrix/tensor bounds.

### 5. `numrs::parallel` — Automatic Parallelism

NumRS automatically uses [Rayon](https://github.com/rayon-rs/rayon) to parallelize operations on large matrices.
- **Threshold**: Operations switch to parallel mode when the number of elements $\ge 1024$.
- **Parallelized Ops**: Addition, Subtraction, Multiplication, Transposition, Hadamard, and Norm.
- **Serial Ops**: Determinant and Inverse (due to the sequential nature of Gaussian elimination).

---

## Design Principles

1.  **Row-Major Storage**: Contiguous memory layout for cache efficiency.
2.  **Zero-Cost Ownership**: Owned operators delegate to reference implementations.
3.  **Predictable Performance**: Automatic parallelism for large workloads, single-threaded for small ones to avoid overhead.
4.  **Type Safety**: `Tensor<T>` is generic, while `Matrix` is specialized for `f64` linear algebra.

---

## Examples

For full code examples of solving linear systems, Markov chains, and more, see [EXAMPLES.md](EXAMPLES.md).
