use crate::matrix::Matrix;

impl Matrix {

    // ── Transpose ─────────────────────────────────────────────────────
    pub fn transpose(&self) -> Matrix {
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix::new(self.cols, self.rows, data)
    }

    // ── Hadamard — element-wise multiplication ────────────────────────
    pub fn hadamard(&self, rhs: &Matrix) -> Matrix {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Hadamard requires equal shapes: ({}×{}) != ({}×{})",
            self.rows, self.cols, rhs.rows, rhs.cols
        );
        let data = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Matrix::new(self.rows, self.cols, data)
    }

    // ── Frobenius Norm — sqrt of sum of squared elements ──────────────
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    // ── Determinant — Gaussian elimination with partial pivoting ──────
    pub fn det(&self) -> f64 {
        assert_eq!(
            self.rows, self.cols,
            "Determinant requires a square matrix, got ({}×{})",
            self.rows, self.cols
        );

        let n = self.rows;
        let mut mat = self.data.clone();
        let mut sign = 1.0_f64;

        for col in 0..n {
            // Find row with largest absolute value in this column
            let pivot_row = (col..n).max_by(|&a, &b| {
                mat[a * n + col].abs()
                    .partial_cmp(&mat[b * n + col].abs())
                    .unwrap()
            });

            let pivot_row = match pivot_row {
                Some(r) if mat[r * n + col].abs() > 1e-10 => r,
                _ => return 0.0, // Singular — no valid pivot
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

        // sign × product of diagonal elements
        sign * (0..n).map(|i| mat[i * n + i]).product::<f64>()
    }

    // ── Inverse — Gauss-Jordan on augmented matrix [A | I] ────────────
    pub fn inverse(&self) -> Option<Matrix> {
        assert_eq!(
            self.rows, self.cols,
            "Inverse requires a square matrix, got ({}×{})",
            self.rows, self.cols
        );

        let n = self.rows;
        let w = 2 * n; // width of augmented matrix

        // Build [A | I]
        let mut aug = vec![0.0; n * w];
        for i in 0..n {
            for j in 0..n { aug[i * w + j] = self.data[i * n + j]; }
            aug[i * w + n + i] = 1.0;
        }

        for col in 0..n {
            // Partial pivoting
            let pivot_row = (col..n).max_by(|&a, &b| {
                aug[a * w + col].abs()
                    .partial_cmp(&aug[b * w + col].abs())
                    .unwrap()
            });

            let pivot_row = match pivot_row {
                Some(r) if aug[r * w + col].abs() > 1e-10 => r,
                _ => return None, // Singular
            };

            if pivot_row != col {
                for k in 0..w { aug.swap(col * w + k, pivot_row * w + k); }
            }

            // Scale pivot row → pivot becomes 1
            let pivot = aug[col * w + col];
            for k in 0..w { aug[col * w + k] /= pivot; }

            // Zero out every other row in this column
            for row in 0..n {
                if row == col { continue; }
                let factor = aug[row * w + col];
                for k in 0..w {
                    let val = aug[col * w + k] * factor;
                    aug[row * w + k] -= val;
                }
            }
        }

        // Extract right half → A⁻¹
        let mut result = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n { result[i * n + j] = aug[i * w + n + j]; }
        }

        Some(Matrix::new(n, n, result))
    }
}
