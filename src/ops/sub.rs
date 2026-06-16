use std::ops::Sub;
use crate::matrix::Matrix;
use crate::error::MatrixError;

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
    /// Subtract two matrices, returning `Err` on shape mismatch.
    pub fn try_sub(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        check_same_shape(self, rhs)?;
        let data = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Matrix::new_unchecked(self.rows, self.cols, data))
    }
}

// ── Operator — panics with MatrixError message on bad input ────────────────
impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Matrix {
        match check_same_shape(self, rhs) {
            Ok(_)  => self.try_sub(rhs).unwrap(),
            Err(e) => panic!("[NumRS] Sub failed: {}", e),
        }
    }
}

impl Sub for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Matrix { &self - &rhs }
}

impl Sub<&Matrix> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Matrix { &self - rhs }
}

impl Sub<Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Matrix { self - &rhs }
}
