# Changelog

All notable changes to NumRS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.8.1] — 2026-07-01

### Added
- `TensorError` enum in `src/error.rs` with four variants:
  `ShapeMismatch`, `RankMismatch`, `InvalidConstruction`,
  `IndexOutOfBounds` — parallel to `MatrixError` but N-dimensional
- `TensorError` re-exported from `lib.rs` as `numrs::TensorError`
- Fallible constructors and accessors on `Tensor<T>`:
  `try_from_vec`, `try_get`, `try_get_mut`
- Safe operation variants: `try_add`, `try_sub`, `try_hadamard`
  (all return `Result<Tensor<T>, TensorError>`)

### Design Notes
- `TensorError` is intentionally separate from `MatrixError`:
  N-dimensional shapes use `Vec<usize>` while Matrix uses `(usize, usize)`;
  merging the two would require dead fields on one side or the other
- Operators (`+`, `-`, `*`, `[]`) still panic ergonomically via
  `try_*` + unwrap, identical to the `Matrix` pattern introduced in [0.6.1]
- Panic messages use the `[NumRS] <operation> failed: <TensorError>` format

---

## [0.8.0] — Tensors & Randomness

### Added
- `src/tensor.rs` — New N-dimensional generic `Tensor<T>` implementation
  - Row-major flat storage with stride-based indexing
  - Generic over any type `T` (f32, f64, i32, etc.)
  - Support for `+`, `-`, scalar `*`, and `.hadamard()`
  - Recursive `Display` implementation for any rank
- **Randomness Support** (for both `Matrix` and `Tensor`)
  - `.rand()` / `.rand_seeded()` — Uniform distribution
  - `.randn()` / `.randn_seeded()` — Standard normal distribution (floats only)
  - Deterministic seeded constructors for reproducible benchmarks and tests
- `rand` and `rand_distr` dependencies

### Changed
- `Matrix::det` and `Matrix::inverse` updated with better internal pivot selection
- Internal `new_unchecked` constructor for `Matrix` to avoid redundant checks in internal math routines

---

## [0.7.0] — Parallel Execution

### Added
- `src/parallel.rs` — `PAR_THRESHOLD` constant (1024 elements) and `should_parallelize(n)`
  helper; single place to tune the serial/parallel cutoff for the whole library
- `rayon = "1.10"` dependency — work-stealing thread pool, zero unsafe code

### Changed — automatic parallelism (no API changes)
All operations below silently switch to `rayon` when the matrix has ≥ 1024 elements;
smaller matrices keep the serial path to avoid thread-spawn overhead.

| Operation | Parallel strategy |
|---|---|
| `+` / `try_add` | `par_iter().zip().map()` |
| `-` / `try_sub` | `par_iter().zip().map()` |
| `*` / `try_mul` | `into_par_iter()` over output rows |
| `scalar *` | `par_iter().map()` |
| `transpose()` | `into_par_iter()` over output columns |
| `hadamard()` | `par_iter().zip().map()` |
| `norm()` | `par_iter().map().sum()` |
| `det()` | unchanged — Gaussian elimination is sequential by nature |
| `inverse()` | unchanged — Gauss-Jordan is sequential by nature |

### Design Notes
- **Zero API changes** — existing code compiles and runs unchanged; speedup is automatic
- **Threshold = 1024** (≈ 32×32): below this, serial is faster due to thread-spawn overhead
- `PAR_THRESHOLD` lives in `parallel.rs` — one constant to tune if needed

---

## [0.6.1] — Error Handling

### Added
- `src/error.rs` — new `MatrixError` enum implementing `std::error::Error` + `Display`
  - `ShapeMismatch { expected, found }` — element-wise ops on incompatible shapes
  - `DimensionMismatch { left_cols, right_rows }` — matrix multiplication dimension error
  - `NonSquare { rows, cols }` — `det()` / `inverse()` called on a non-square matrix
  - `NonInvertible` — `inverse()` failed because the matrix is singular
  - `InvalidConstruction { rows, cols, data_length }` — `Matrix::new()` size mismatch
  - `IndexOutOfBounds { index, dimensions }` — `[(i,j)]` access beyond bounds
- `MatrixError` re-exported from `lib.rs` — `use numrs::MatrixError;`

### Changed
- **`det()`** — now returns `Result<f64, MatrixError>` instead of panicking on non-square input
- **`inverse()`** — now returns `Result<Matrix, MatrixError>` replacing `Option<Matrix>`;
  singular matrices produce `Err(MatrixError::NonInvertible)`
- **`hadamard()`** — now returns `Result<Matrix, MatrixError>` instead of panicking
- **Operators `+` `-` `*`** — panics replaced with clean `[NumRS] <MatrixError message>` format
- **`Matrix::new()`** — panic message now uses `MatrixError::InvalidConstruction`
- **`Index` / `IndexMut`** — panic message now uses `MatrixError::IndexOutOfBounds`

### Added (safe variants)
- `Matrix::try_add(&rhs)` → `Result<Matrix, MatrixError>`
- `Matrix::try_sub(&rhs)` → `Result<Matrix, MatrixError>`
- `Matrix::try_mul(&rhs)` → `Result<Matrix, MatrixError>`

### Internal
- `Matrix::new_unchecked(rows, cols, data)` — `pub(crate)` constructor that skips the
  length assertion; used internally wherever the shape is already validated.
  Includes a `debug_assert` to catch bugs in debug builds.

### Design Notes
- **Hybrid approach**: operators (`+`, `-`, `*`) still panic for ergonomic math syntax;
  `try_*` variants provide `Result`-returning alternatives for safe contexts
- All panics now route through `MatrixError::Display` — consistent, readable messages
- `det()` returning `Ok(0.0)` for singular matrices is intentional:
  a determinant of zero is a valid mathematical result, not an error

---

## [0.6.0] — Mathematical Operations

### Added
- `transpose()` — swap rows and columns; returns new `Matrix` with flipped dimensions
- `hadamard(&b)` — element-wise multiplication; asserts equal shapes
- `norm()` — Frobenius norm: √(sum of all squared elements)
- `det()` — determinant via Gaussian elimination with partial pivoting; returns `0.0` for singular matrices
- `inverse()` — Gauss-Jordan on augmented matrix `[A | I]`; returns `Option<Matrix>` (`None` if singular)
- New module: `src/math.rs` — all mathematical methods live here as `impl Matrix`

### Notes
- `det()` uses partial pivoting (largest absolute value per column) for numerical stability
- `inverse()` returns `None` instead of panicking on singular input — idiomatic Rust error handling

---

## [0.5.0] — Constructors & Indexing

### Added
- `Matrix::fill(r, c, v)` — base constructor, fills matrix with a given `f64` value
- `Matrix::zeros(r, c)` — delegates to `fill` with `0.0`
- `Matrix::ones(r, c)` — delegates to `fill` with `1.0`
- `Matrix::eye(n)` — n×n identity matrix; zeros except main diagonal
- `Index<(usize, usize)>` trait — read access: `matrix[(i, j)]`
- `IndexMut<(usize, usize)>` trait — write access: `matrix[(i, j)] = v`
- New file: `src/ops/index.rs`

### Changed
- `Matrix::new` now validates that `rows × cols == data.len()` with a descriptive panic message
- `zeros` and `ones` delegate to `fill` — single source of truth for initialization logic

---

## [0.4.0] — Stable Operators & Sub

### Added
- All arithmetic operators now implemented for all 4 ownership combinations:
  `T op T`, `T op &T`, `&T op T`, `&T op &T` — no more forced `.clone()`
- `Sub` trait for matrix subtraction with shape assertion
- `Mul<f64>` — scalar multiplication: `&matrix * 2.0` and `matrix * 2.0`
- `Mul<Matrix> for f64` — commutative scalar: `2.0 * &matrix` and `2.0 * matrix`
- New files: `src/ops/sub.rs`, `src/ops/scalar.rs`

### Fixed
- `Add` and `Mul` previously consumed ownership — reference-based implementations are now primary;
  owned versions delegate to them

### Design
- One place for logic (`&Matrix op &Matrix`), all other combinations are one-liners that delegate

---

## [0.3.0] — Module Restructure

### Changed
- Split single-file structure into organized modules
- `src/matrix.rs` — Matrix struct, Display trait, `ns_array!` macro
- `src/ops/` — all operator trait implementations
- `src/ops/add.rs` — Add trait
- `src/ops/mul.rs` — Mul trait
- `src/ops/mod.rs` — re-exports submodules
- `src/main.rs` — entry point only, no library logic

### Notes
- `ns_array!` macro uses `$crate::matrix::Matrix` for correct path resolution across modules

---

## [0.2.0] — Colored Display

### Added
- `Display` trait implementation — replaces raw `{:?}` debug output
- Per-column width alignment — each column sized to its widest element
- Color coding via `colored` crate:
  - Brackets `┌ │ └` — cyan bold
  - Negative values — yellow
  - Zero values — dimmed
  - Positive values — white
- Single-row matrices use `[ ]` brackets instead of box-drawing characters
- Header line: `Matrix (rows×cols)` in bold

### Dependencies
- Added `colored = "2"` to `Cargo.toml`

---

## [0.1.0] — Initial Release

### Added
- `Matrix` struct with fields: `rows: usize`, `cols: usize`, `data: Vec<f64>`
- `Matrix::new(rows, cols, data)` constructor
- `ns_array!` macro — create matrices with Python-like syntax:
  ```rust
  let a = ns_array![[1, 2, 3], [4, 5, 6]];
  ```
  Macro validates that all rows have equal length at compile-expand time
- `Add` trait — matrix addition with dimension assertion
- `Mul` trait — matrix multiplication (dot product) with dimension assertion
- Internal row-major storage: element `(i, j)` at index `i * cols + j`
