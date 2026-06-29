use std::fmt;

/// All errors that NumRS `Matrix` operations can produce.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixError {
    /// Element-wise ops (add, sub, hadamard) require identical shapes.
    ShapeMismatch {
        expected: (usize, usize),
        found:    (usize, usize),
    },

    /// Matrix multiplication requires left.cols == right.rows.
    DimensionMismatch {
        left_cols:  usize,
        right_rows: usize,
    },

    /// det() and inverse() require a square matrix.
    NonSquare {
        rows: usize,
        cols: usize,
    },

    /// inverse() failed because the matrix is singular (det ≈ 0).
    NonInvertible,

    /// Matrix::new() received data whose length ≠ rows × cols.
    InvalidConstruction {
        rows:        usize,
        cols:        usize,
        data_length: usize,
    },

    /// Index access was outside the matrix bounds.
    IndexOutOfBounds {
        index:      (usize, usize),
        dimensions: (usize, usize),
    },
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::ShapeMismatch { expected, found } => write!(
                f,
                "shape mismatch: expected ({}×{}), found ({}×{})",
                expected.0, expected.1, found.0, found.1
            ),

            MatrixError::DimensionMismatch { left_cols, right_rows } => write!(
                f,
                "dimension mismatch for multiplication: \
                 left matrix has {} columns but right matrix has {} rows",
                left_cols, right_rows
            ),

            MatrixError::NonSquare { rows, cols } => write!(
                f,
                "operation requires a square matrix, got ({}×{})",
                rows, cols
            ),

            MatrixError::NonInvertible => write!(
                f,
                "matrix is singular (determinant ≈ 0) and cannot be inverted"
            ),

            MatrixError::InvalidConstruction { rows, cols, data_length } => write!(
                f,
                "invalid construction: {}×{} matrix requires {} elements, got {}",
                rows, cols, rows * cols, data_length
            ),

            MatrixError::IndexOutOfBounds { index, dimensions } => write!(
                f,
                "index ({}, {}) is out of bounds for matrix ({}×{})",
                index.0, index.1, dimensions.0, dimensions.1
            ),
        }
    }
}

// Integrates with Rust's standard error ecosystem
impl std::error::Error for MatrixError {}

/// All errors that NumRS `Tensor<T>` operations can produce.
///
/// Kept deliberately separate from `MatrixError`: `Tensor` is N-dimensional
/// and generic over `T`, so its failure modes (rank mismatch, N-D shape
/// mismatch) don't map cleanly onto `Matrix`'s 2D-specific variants
/// (`DimensionMismatch`, `NonSquare`, ...). Merging the two would force
/// either dead fields on `Matrix`'s variants or `Vec<usize>` shapes on
/// `Matrix`'s 2D-only ones — this keeps each error type honest about what
/// its own type can actually produce.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    /// Element-wise ops (add, sub, hadamard) require identical shapes.
    ShapeMismatch {
        expected: Vec<usize>,
        found:    Vec<usize>,
    },

    /// An indexing operation supplied a different number of indices
    /// than the tensor's rank (`shape.len()`).
    RankMismatch {
        expected_rank: usize,
        found_rank:    usize,
    },

    /// `Tensor::from_vec()` received data whose length didn't match the
    /// product of the requested shape.
    InvalidConstruction {
        shape:       Vec<usize>,
        expected_len: usize,
        found_len:   usize,
    },

    /// Index access was outside the tensor's bounds along some axis.
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, found } => write!(
                f,
                "shape mismatch: expected {:?}, found {:?}",
                expected, found
            ),

            TensorError::RankMismatch { expected_rank, found_rank } => write!(
                f,
                "rank mismatch: tensor has rank {} but index has rank {}",
                expected_rank, found_rank
            ),

            TensorError::InvalidConstruction { shape, expected_len, found_len } => write!(
                f,
                "invalid construction: shape {:?} requires {} elements, got {}",
                shape, expected_len, found_len
            ),

            TensorError::IndexOutOfBounds { index, shape } => write!(
                f,
                "index {:?} is out of bounds for tensor with shape {:?}",
                index, shape
            ),
        }
    }
}

impl std::error::Error for TensorError {}
