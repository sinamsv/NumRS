use std::ops::Add;
use rayon::prelude::*;
use crate::matrix::Matrix;
use crate::error::MatrixError;
use crate::parallel::should_parallelize;

// ── Internal shape-check helper ────────────────────────────────────────────
fn check_same_shape(lhs: &Matrix, rhs: &Matrix) -> Result<(), MatrixError> {
    if lhs.rows == rhs.rows && lhs.cols == rhs.cols {
        Ok(())
    } else {
        Err(MatrixError::ShapeMismatch {
            expected: (lhs.rows, lhs.cols),
            found:    (rhs.rows, rhs.cols),
        })
    }
}

// ── Safe version ───────────────────────────────────────────────────────────
impl Matrix {
    /// Add two matrices, returning `Err` on shape mismatch.
    /// Automatically uses parallel execution for large matrices.
    pub fn try_add(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        check_same_shape(self, rhs)?;
        let data = if should_parallelize(self.data.len()) {
            self.data.par_iter()
                .zip(rhs.data.par_iter())
                .map(|(a, b)| a + b)
                .collect()
        } else {
            self.data.iter()
                .zip(rhs.data.iter())
                .map(|(a, b)| a + b)
                .collect()
        };
        Ok(Matrix::new_unchecked(self.rows, self.cols, data))
    }
}

// ── Operator — panics with MatrixError message on bad input ────────────────
impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
        match check_same_shape(self, rhs) {
            Ok(_)  => self.try_add(rhs).unwrap(),
            Err(e) => panic!("[NumRS] Add failed: {}", e),
        }
    }
}

impl Add for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Matrix { &self + &rhs }
}

impl Add<&Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Matrix { &self + rhs }
}

impl Add<Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Matrix { self + &rhs }
}
