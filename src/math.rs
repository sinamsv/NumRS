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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ns_array;
    use crate::assert_close;

    // ── transpose ───────────────────────────────────────────────────────
    #[test]
    fn transpose_swaps_rows_and_cols() {
        let a = ns_array![[1, 2, 3], [4, 5, 6]]; // 2×3
        let t = a.transpose();
        assert_eq!((t.rows, t.cols), (3, 2));
        // t[(j,i)] should equal a[(i,j)]
        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_twice_returns_original() {
        let a = ns_array![[1, 2, 3], [4, 5, 6]];
        let tt = a.transpose().transpose();
        assert_eq!(tt.data, a.data);
        assert_eq!((tt.rows, tt.cols), (a.rows, a.cols));
    }

    // ── hadamard ────────────────────────────────────────────────────────
    #[test]
    fn hadamard_multiplies_elementwise() {
        let a = ns_array![[1, 2], [3, 4]];
        let b = ns_array![[5, 6], [7, 8]];
        let h = a.hadamard(&b).unwrap();
        assert_eq!(h.data, vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn hadamard_returns_shape_mismatch_error() {
        let a = ns_array![[1, 2], [3, 4]];   // 2×2
        let b = ns_array![[1, 2, 3]];        // 1×3
        match a.hadamard(&b) {
            Err(MatrixError::ShapeMismatch { .. }) => {}
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    // ── norm ────────────────────────────────────────────────────────────
    #[test]
    fn norm_computes_frobenius_norm() {
        // [3, 4] → sqrt(9+16) = 5
        let v = ns_array![[3, 4]];
        assert_close!(v.norm(), 5.0);
    }

    #[test]
    fn norm_of_zero_matrix_is_zero() {
        let z = Matrix::zeros(3, 3);
        assert_close!(z.norm(), 0.0);
    }

    // ── det ─────────────────────────────────────────────────────────────
    #[test]
    fn det_of_2x2_matches_known_value() {
        // det([[3,1],[2,4]]) = 3*4 - 1*2 = 10
        let m = ns_array![[3, 1], [2, 4]];
        assert_close!(m.det().unwrap(), 10.0);
    }

    #[test]
    fn det_of_identity_is_one() {
        let eye = Matrix::eye(4);
        assert_close!(eye.det().unwrap(), 1.0);
    }

    #[test]
    fn det_of_singular_matrix_is_zero() {
        // Row 2 = 2 * Row 1 → linearly dependent → singular
        let m = ns_array![[1, 2], [2, 4]];
        assert_close!(m.det().unwrap(), 0.0);
    }

    #[test]
    fn det_returns_non_square_error() {
        let m = ns_array![[1, 2, 3], [4, 5, 6]]; // 2×3
        match m.det() {
            Err(MatrixError::NonSquare { rows, cols }) => {
                assert_eq!((rows, cols), (2, 3));
            }
            other => panic!("expected NonSquare, got {:?}", other),
        }
    }

    // ── inverse ─────────────────────────────────────────────────────────
    #[test]
    fn inverse_of_2x2_matches_known_value() {
        // inverse([[3,1],[2,4]]) = (1/10) * [[4,-1],[-2,3]]
        let m = ns_array![[3, 1], [2, 4]];
        let inv = m.inverse().unwrap();
        assert_close!(inv[(0, 0)], 0.4);
        assert_close!(inv[(0, 1)], -0.1);
        assert_close!(inv[(1, 0)], -0.2);
        assert_close!(inv[(1, 1)], 0.3);
    }

    #[test]
    fn inverse_of_identity_is_identity() {
        let eye = Matrix::eye(3);
        let inv = eye.inverse().unwrap();
        assert_eq!(inv.data, eye.data);
    }

    #[test]
    fn inverse_of_singular_matrix_is_non_invertible_error() {
        let m = ns_array![[1, 2], [2, 4]]; // singular
        match m.inverse() {
            Err(MatrixError::NonInvertible) => {}
            other => panic!("expected NonInvertible, got {:?}", other),
        }
    }

    #[test]
    fn inverse_returns_non_square_error() {
        let m = ns_array![[1, 2, 3], [4, 5, 6]]; // 2×3
        match m.inverse() {
            Err(MatrixError::NonSquare { .. }) => {}
            other => panic!("expected NonSquare, got {:?}", other),
        }
    }

    #[test]
    fn matrix_times_its_inverse_is_identity() {
        let m = ns_array![[3, 1], [2, 4]];
        let inv = m.inverse().unwrap();
        let product = &m * &inv;

        let eye = Matrix::eye(2);
        for i in 0..2 {
            for j in 0..2 {
                assert_close!(product[(i, j)], eye[(i, j)], 1e-9);
            }
        }
    }
}
