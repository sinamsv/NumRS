use std::ops::Mul;
use rayon::prelude::*;
use crate::matrix::Matrix;
use crate::error::MatrixError;
use crate::parallel::should_parallelize;

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
    /// For large matrices each output row is computed on a separate thread.
    pub fn try_mul(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        check_mul_dims(self, rhs)?;

        // Total output elements: self.rows × rhs.cols
        let out_size = self.rows * rhs.cols;

        let data = if should_parallelize(out_size) {
            // Each row of the result is independent — split on rows
            (0..self.rows).into_par_iter().flat_map(|i| {
                (0..rhs.cols).map(move |j| {
                    (0..self.cols)
                        .map(|k| self.data[i * self.cols + k] * rhs.data[k * rhs.cols + j])
                        .sum::<f64>()
                }).collect::<Vec<f64>>()
            }).collect()
        } else {
            let mut data = vec![0.0; out_size];
            for i in 0..self.rows {
                for j in 0..rhs.cols {
                    for k in 0..self.cols {
                        data[i * rhs.cols + j] +=
                            self.data[i * self.cols + k] * rhs.data[k * rhs.cols + j];
                    }
                }
            }
            data
        };

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ns_array;

    #[test]
    fn mul_produces_correct_dot_product() {
        let a = ns_array![[1, 2], [3, 4]];
        let b = ns_array![[5, 6], [7, 8]];
        let c = &a * &b;
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn mul_by_identity_returns_same_matrix() {
        let a = ns_array![[1, 2, 3], [4, 5, 6]];
        let eye = Matrix::eye(3);
        let result = &a * &eye;
        assert_eq!(result.data, a.data);
        assert_eq!((result.rows, result.cols), (a.rows, a.cols));
    }

    #[test]
    fn mul_result_has_correct_shape() {
        // (2×3) * (3×4) = (2×4)
        let a = Matrix::zeros(2, 3);
        let b = Matrix::zeros(3, 4);
        let c = &a * &b;
        assert_eq!((c.rows, c.cols), (2, 4));
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn mul_panics_on_dimension_mismatch() {
        let a = ns_array![[1, 2, 3]];     // 1×3
        let b = ns_array![[1, 2, 3]];     // 1×3 — cols(3) != rows(1)
        let _ = &a * &b;
    }

    #[test]
    fn try_mul_returns_dimension_mismatch_error() {
        let a = Matrix::zeros(2, 3); // cols = 3
        let b = Matrix::zeros(2, 2); // rows = 2 — mismatch

        match a.try_mul(&b) {
            Err(MatrixError::DimensionMismatch { left_cols, right_rows }) => {
                assert_eq!(left_cols, 3);
                assert_eq!(right_rows, 2);
            }
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    #[test]
    fn try_mul_parallel_matches_serial_for_known_result() {
        // 40×40 identity-like check above PAR_THRESHOLD (1024 elements)
        let n = 40;
        let a_data: Vec<f64> = (0..n * n).map(|i| (i % 7) as f64).collect();
        let a = Matrix::new(n, n, a_data);
        let eye = Matrix::eye(n);

        // A × I should always equal A, regardless of which code path
        // (serial vs. rayon) was used internally.
        let result = a.try_mul(&eye).unwrap();
        assert_eq!(result.data, a.data);
    }
}
