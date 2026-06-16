use rayon::prelude::*;
use crate::matrix::Matrix;
use crate::error::MatrixError;
use crate::parallel::should_parallelize;

impl Matrix {

    // ── Transpose ─────────────────────────────────────────────────────────
    pub fn transpose(&self) -> Matrix {
        let rows = self.rows;
        let cols = self.cols;

        // Each output element [j][i] = input[i][j] — all independent
        let data = if should_parallelize(rows * cols) {
            (0..cols).into_par_iter().flat_map(|j| {
                (0..rows).map(move |i| self.data[i * cols + j])
                    .collect::<Vec<f64>>()
            }).collect()
        } else {
            let mut data = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    data[j * rows + i] = self.data[i * cols + j];
                }
            }
            data
        };

        Matrix::new_unchecked(cols, rows, data)
    }

    // ── Hadamard — element-wise multiplication ────────────────────────────
    pub fn hadamard(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(MatrixError::ShapeMismatch {
                expected: (self.rows, self.cols),
                found:    (rhs.rows,  rhs.cols),
            });
        }
        let data = if should_parallelize(self.data.len()) {
            self.data.par_iter()
                .zip(rhs.data.par_iter())
                .map(|(a, b)| a * b)
                .collect()
        } else {
            self.data.iter()
                .zip(rhs.data.iter())
                .map(|(a, b)| a * b)
                .collect()
        };
        Ok(Matrix::new_unchecked(self.rows, self.cols, data))
    }

    // ── Frobenius Norm ─────────────────────────────────────────────────────
    pub fn norm(&self) -> f64 {
        let sum_sq = if should_parallelize(self.data.len()) {
            self.data.par_iter().map(|&x| x * x).sum::<f64>()
        } else {
            self.data.iter().map(|&x| x * x).sum::<f64>()
        };
        sum_sq.sqrt()
    }

    // ── Determinant — Gaussian elimination (sequential by nature) ─────────
    pub fn det(&self) -> Result<f64, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NonSquare { rows: self.rows, cols: self.cols });
        }

        let n = self.rows;
        let mut mat = self.data.clone();
        let mut sign = 1.0_f64;

        for col in 0..n {
            let pivot_row = (col..n).max_by(|&a, &b| {
                mat[a * n + col].abs()
                    .partial_cmp(&mat[b * n + col].abs())
                    .unwrap()
            });

            let pivot_row = match pivot_row {
                Some(r) if mat[r * n + col].abs() > 1e-10 => r,
                _ => return Ok(0.0),
            };

            if pivot_row != col {
                for k in 0..n { mat.swap(col * n + k, pivot_row * n + k); }
                sign *= -1.0;
            }

            let pivot = mat[col * n + col];
            for row in (col + 1)..n {
                let factor = mat[row * n + col] / pivot;
                for k in col..n {
                    let val = mat[col * n + k] * factor;
                    mat[row * n + k] -= val;
                }
            }
        }

        Ok(sign * (0..n).map(|i| mat[i * n + i]).product::<f64>())
    }

    // ── Inverse — Gauss-Jordan (sequential by nature) ─────────────────────
    pub fn inverse(&self) -> Result<Matrix, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NonSquare { rows: self.rows, cols: self.cols });
        }

        let n = self.rows;
        let w = 2 * n;

        let mut aug = vec![0.0; n * w];
        for i in 0..n {
            for j in 0..n { aug[i * w + j] = self.data[i * n + j]; }
            aug[i * w + n + i] = 1.0;
        }

        for col in 0..n {
            let pivot_row = (col..n).max_by(|&a, &b| {
                aug[a * w + col].abs()
                    .partial_cmp(&aug[b * w + col].abs())
                    .unwrap()
            });

            let pivot_row = match pivot_row {
                Some(r) if aug[r * w + col].abs() > 1e-10 => r,
                _ => return Err(MatrixError::NonInvertible),
            };

            if pivot_row != col {
                for k in 0..w { aug.swap(col * w + k, pivot_row * w + k); }
            }

            let pivot = aug[col * w + col];
            for k in 0..w { aug[col * w + k] /= pivot; }

            for row in 0..n {
                if row == col { continue; }
                let factor = aug[row * w + col];
                for k in 0..w {
                    let val = aug[col * w + k] * factor;
                    aug[row * w + k] -= val;
                }
            }
        }

        let mut result = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n { result[i * n + j] = aug[i * w + n + j]; }
        }

        Ok(Matrix::new_unchecked(n, n, result))
    }
}
