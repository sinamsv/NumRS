# NumRS — Usage Guide

This guide explains everything you need to know to use NumRS: what each
constructor, operator, and method does, how the two core types relate to
each other, and what isn't built yet. Every example below was run against
the actual source to confirm it compiles and produces the output shown.

---

## What is NumRS?

NumRS is a linear algebra and tensor library built from scratch in Rust,
with two core types:

| Type        | Dimensions | Element type | What it's for                                   |
|-------------|------------|---------------|--------------------------------------------------|
| `Matrix`    | 2D only    | `f64` only    | Linear algebra: solving systems, transforms, crypto math |
| `Tensor<T>` | N-D, any rank | generic `T` | N-dimensional data — the basis for future AI/tensor work |

Both share the same underlying idea: a single flat `Vec` holding the data
in **row-major** order, instead of nested `Vec<Vec<...>>`. This is faster
(one heap allocation, cache-friendly) and is the standard layout used by
NumPy, ndarray, and most numerical libraries.

`Matrix` and `Tensor<T>` are **independent types** — `Tensor` was added
without changing a single line of `Matrix`'s public API.

---

## Installation

NumRS isn't published to crates.io yet. Use it as a path or git dependency:

```toml
[dependencies]
numrs = { path = "../NumRS" }
# or: numrs = { git = "https://github.com/sinamsv/NumRS.git" }
```

Its own dependencies (already in `Cargo.toml`) are:

| Crate        | Why it's there                                              |
|--------------|--------------------------------------------------------------|
| `colored`    | Colored terminal output for `Display` (both `Matrix` and `Tensor`) |
| `num-traits` | Gives `Tensor<T>` generic access to `T::zero()` / `T::one()` so it isn't locked to `f64` |

---

## Quick Start

```rust
use numrs::{Matrix, Tensor, ns_array};

fn main() {
    // A 2D matrix
    let a = ns_array![[1, 2, 3], [4, 5, 6]];
    println!("{}", a);

    // A 3D tensor of i64
    let t: Tensor<i64> = Tensor::zeros(&[2, 2, 2]);
    println!("{}", t);
}
```

---

## Part 1 — `Matrix`

`Matrix` is a 2D, `f64`-only type with a full set of linear algebra
operations. Internally, element `(i, j)` lives at `data[i * cols + j]`.

### 1.1 Creating a Matrix

| Call                              | Result                                              |
|------------------------------------|------------------------------------------------------|
| `ns_array![[1,2],[3,4]]`          | Matrix from literal rows, Python-like syntax        |
| `Matrix::new(rows, cols, data)`   | Matrix from an explicit flat `Vec<f64>` — panics if `rows*cols != data.len()` |
| `Matrix::zeros(rows, cols)`       | All elements `0.0`                                  |
| `Matrix::ones(rows, cols)`        | All elements `1.0`                                  |
| `Matrix::fill(rows, cols, v)`     | All elements set to `v`                             |
| `Matrix::eye(n)`                  | n×n identity matrix                                 |

```rust
let a = ns_array![[1, 2, 3], [4, 5, 6]]; // 2×3
let z = Matrix::zeros(2, 3);
let id = Matrix::eye(3);
```

### 1.2 Reading and writing elements

```rust
let mut m = Matrix::zeros(2, 2);
m[(0, 1)] = 5.0;          // write
let v = m[(0, 1)];        // read -> 5.0
```
Implemented via `Index`/`IndexMut` on the tuple `(row, col)`. Out-of-bounds
access panics with a descriptive message.

### 1.3 Arithmetic operators

| Expression  | Meaning                                                        |
|-------------|------------------------------------------------------------------|
| `&a + &b`   | Element-wise addition — shapes must match                       |
| `&a - &b`   | Element-wise subtraction — shapes must match                    |
| `&a * &b`   | **Matrix multiplication** (dot product) — `a.cols` must equal `b.rows` |
| `&a * 2.0`  | Scalar multiplication                                            |
| `2.0 * &a`  | Scalar multiplication, commutative form                          |

Every operator above also works in **owned form** (`a + b`, `a * b`, ...),
and in all four combinations of owned/reference. References are the
primary implementation; owned versions just delegate to them so you never
pay for a hidden `.clone()`.

```rust
let sum    = &a + &a;
let scaled = &a * 2.0;
let same   = 2.0 * &a;       // commutative — same result as above
```

### 1.4 Math operations

| Method            | Returns          | Description                                          |
|--------------------|-------------------|--------------------------------------------------------|
| `.transpose()`     | `Matrix`          | Swaps rows and columns                                |
| `.hadamard(&b)`    | `Matrix`          | Element-wise multiplication — shapes must match exactly (different from `*`, which is matrix multiplication) |
| `.norm()`          | `f64`             | Frobenius norm: √(sum of all squared elements)         |
| `.det()`           | `f64`             | Determinant via Gaussian elimination with partial pivoting. Panics if not square; returns `0.0` if singular |
| `.inverse()`       | `Option<Matrix>`  | Gauss-Jordan inverse. Panics if not square; `None` if singular (no panic) |

```rust
let m = ns_array![[3, 1], [2, 4]];
let det = m.det();                 // 10.0
let inv = m.inverse();             // Some(Matrix)
```

### 1.5 Printing a Matrix

`Matrix` implements `Display`, so `println!("{}", m)` renders a colored,
column-aligned table:
- Cyan box-drawing brackets (`┌ │ └`, or `[ ]` for a single row)
- Negative values in yellow, zeros dimmed, positive values white
- Each column is sized to its widest value, values shown to 3 decimals

---

## Part 2 — `Tensor<T>`

`Tensor<T>` generalizes the same row-major idea to **any rank** (0, 1, 2,
3, ... dimensions) and **any numeric type** `T` (e.g. `f32`, `f64`, `i32`,
`i64`, `u32`, ...) — not just `f64`.

### 2.1 Shape, strides, and rank

A `Tensor` doesn't store `rows`/`cols`; it stores a `shape: Vec<usize>`
(one entry per dimension) and a matching `strides: Vec<usize>` that tells
you how far to jump in the flat data for each dimension. For shape
`[2, 3, 4]`:

```
shape   = [2, 3, 4]
strides = [12, 4, 1]     // 4 = 4*1 (size of last dim), 12 = 3*4
```

This is exactly the `i * cols + j` scheme `Matrix` uses, just extended to
N dimensions. `rank()` is simply `shape.len()` — how many dimensions the
tensor has.

### 2.2 Creating a Tensor

| Call                                 | Requires       | Result                              |
|---------------------------------------|----------------|---------------------------------------|
| `Tensor::from_vec(shape, data)`      | —              | Tensor from explicit shape + flat data; panics if sizes don't match |
| `Tensor::zeros(shape)`               | `T: Zero`      | All elements `T::zero()`              |
| `Tensor::ones(shape)`                | `T: One`       | All elements `T::one()`               |
| `Tensor::fill(shape, value)`         | `T: Clone`     | All elements set to `value`           |

```rust
let t:  Tensor<f64> = Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let tz: Tensor<f64> = Tensor::zeros(&[2, 2, 2]);   // a 3D tensor
let to: Tensor<i64> = Tensor::ones(&[3]);          // a 1D tensor of i64
```

`Zero`/`One` come from the `num-traits` crate and are already implemented
for every Rust numeric primitive, so `zeros`/`ones` work out of the box
for `f32`, `f64`, `i8`...`i128`, `u8`...`u128`, `isize`, `usize`.

### 2.3 Inspecting a Tensor

| Method          | Returns      | Meaning                                |
|------------------|--------------|------------------------------------------|
| `.rank()`       | `usize`      | Number of dimensions                    |
| `.shape()`      | `&[usize]`   | Size of each dimension                  |
| `.strides()`    | `&[usize]`   | Flat-index jump per dimension           |
| `.len()`        | `usize`      | Total number of elements                |
| `.is_empty()`   | `bool`       | `true` if there are no elements         |
| `.as_slice()`   | `&[T]`       | The raw underlying flat data            |

### 2.4 Reading and writing elements

```rust
let mut t: Tensor<f64> = Tensor::zeros(&[2, 2]);
t[&[1, 1]] = 42.0;        // write
let v = t[&[1, 1]];       // read -> 42.0
```
The index is a slice with **one number per dimension** — `&[1, 1]` for a
2D tensor, `&[0, 2, 1]` for a 3D one, and so on. Array literals like
`&[1, 1]` coerce to `&[usize]` automatically. Passing the wrong number of
indices, or an out-of-bounds one, panics with a descriptive message.

### 2.5 Arithmetic operators

| Expression       | Requires                  | Meaning                                          |
|-------------------|---------------------------|----------------------------------------------------|
| `&a + &b`        | `T: Copy + Add`           | Element-wise addition — shapes must match exactly  |
| `&a - &b`        | `T: Copy + Sub`           | Element-wise subtraction — shapes must match exactly |
| `&a * scalar`    | `T: Copy + Mul`           | Scalar multiplication                              |
| `a.hadamard(&b)` | `T: Copy + Mul`           | Element-wise multiplication — shapes must match exactly |

`+` and `-` also work in owned form and all four ownership combinations,
same as `Matrix`. **One asymmetry to know about:** scalar multiplication
currently only works as `tensor * scalar`, not `scalar * tensor` — unlike
`Matrix`, the commutative form (`2.0 * &t`) isn't implemented yet.

```rust
let a: Tensor<i64> = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
let b: Tensor<i64> = Tensor::ones(&[2, 2]);
let sum  = &a + &b;          // [[2,3],[4,5]]
let prod = &a * 10;          // [[10,20],[30,40]] — works
// let bad = 10 * &a;        // does NOT compile yet
```

There is no `*` between two tensors (no matrix-multiplication or
tensor-contraction equivalent yet) and no `det`/`inverse`/`norm` — those
remain `Matrix`-only for now.

### 2.6 Which types can I use?

Any `T` that implements the traits a given operation needs. In practice,
every signed/unsigned integer and float primitive works for everything
except `Display`-based printing of negative-aware coloring, which needs
`PartialOrd` (also true for all of them). A few examples that work today:
`f32`, `f64`, `i32`, `i64`, `u32`, `u64`.

### 2.7 Printing a Tensor

`Tensor<T>` implements `Display` recursively, so it works for **any
rank** — unlike a fixed 2D table. A 3D tensor of shape `[2,2,2]` prints
as nested brackets:

```
Tensor [2, 2, 2]
[
  [
    [0, 0],
    [0, 0]
  ],
  [
    [0, 0],
    [0, 0]
  ]
]
```

Same coloring rule as `Matrix` (negative → yellow, zero → dimmed,
positive → white). **One difference from `Matrix` to know about:**
because `T` is generic, `Display` can't assume `{:.3}` formatting —
numbers print using `T`'s own natural formatting (e.g. `4.5`, not
`4.500`), and columns are not padded/aligned the way `Matrix`'s are.

---

## Shared design principles

- **Row-major flat storage** for both types — no nested `Vec`s, one
  contiguous allocation.
- **Reference-first operators** — `&Matrix op &Matrix` / `&Tensor op
  &Tensor` hold the real logic; every owned variant just delegates, so
  you choose owned vs. borrowed based on whether you need the original
  afterward, never because of a performance penalty.
- **Panics, not `Result`, for invalid input** (shape mismatches,
  out-of-bounds indices, non-square matrices for `det`/`inverse`). This
  is a known gap — see below.
- **`Tensor` never touches `Matrix`'s code.** They're separate types on
  purpose, so adding `Tensor` couldn't break anything that already
  depended on `Matrix`.

---

## Known limitations (today)

- No custom error type — invalid shapes/indices panic instead of
  returning a `Result`.
- No parallelism — every operation is single-threaded.
- `Tensor` has no `reshape`, `permute` (the N-D equivalent of
  `transpose`), broadcasting, or tensor contraction/matmul yet.
- `Tensor` has no `det`, `inverse`, or `norm` — those exist only on
  `Matrix`.
- `Tensor` scalar multiplication is one-directional (`tensor * scalar`
  only).
- `Tensor`'s `Display` isn't column-aligned like `Matrix`'s.
- No conversions between `Matrix` and `Tensor` yet.
- No LU/QR decomposition, eigenvalues, or GPU acceleration yet.

## What's next

See `README.md` for the full roadmap. The current focus after `Tensor<T>`
is adding shape-manipulation methods (`reshape`, `permute`) before moving
on to decompositions and hardware acceleration.
