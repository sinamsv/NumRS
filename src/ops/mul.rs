use std::ops::Mul;
use crate::matrix::Matrix;
use crate::error::MatrixError;

// ── Internal dimension-check helper ───────────────────────────────────────
fn check_mul_dims(lhs: &Matrix, rhs: &Matrix) -> Result<(), MatrixError> {
    if lhs.cols == rhs.rows {
        Ok(())
    } else {
        Err(MatrixError::DimensionMismatch {
            left_cols:  lhs.cols,
            right_rows: rhs.rows,
        })
    }
}

// ── Safe version ───────────────────────────────────────────────────────────
impl Matrix {
    /// Multiply two matrices (dot product), returning `Err` on dimension mismatch.
    pub fn try_mul(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        check_mul_dims(self, rhs)?;
        let mut data = vec![0.0; self.rows * rhs.cols];
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                for k in 0..self.cols {
                    data[i * rhs.cols + j] +=
                        self.data[i * self.cols + k] * rhs.data[k * rhs.cols + j];
                }
            }
        }
        Ok(Matrix::new_unchecked(self.rows, rhs.cols, data))
    }
}

// ── Operator — panics with MatrixError message on bad input ────────────────
impl Mul for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        match check_mul_dims(self, rhs) {
            Ok(_)  => self.try_mul(rhs).unwrap(),
            Err(e) => panic!("[NumRS] Mul failed: {}", e),
        }
    }
}

impl Mul for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Matrix { &self * &rhs }
}

impl Mul<&Matrix> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Matrix { &self * rhs }
}

impl Mul<Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Matrix { self * &rhs }
}
