use std::ops::Mul;
use rayon::prelude::*;
use crate::matrix::Matrix;
use crate::parallel::should_parallelize;

// &Matrix * f64
impl Mul<f64> for &Matrix {
    type Output = Matrix;
    fn mul(self, scalar: f64) -> Matrix {
        let data = if should_parallelize(self.data.len()) {
            self.data.par_iter().map(|&x| x * scalar).collect()
        } else {
            self.data.iter().map(|&x| x * scalar).collect()
        };
        Matrix::new_unchecked(self.rows, self.cols, data)
    }
}

// Matrix * f64  →  delegates
impl Mul<f64> for Matrix {
    type Output = Matrix;
    fn mul(self, scalar: f64) -> Matrix { &self * scalar }
}

// f64 * &Matrix
impl Mul<&Matrix> for f64 {
    type Output = Matrix;
    fn mul(self, matrix: &Matrix) -> Matrix { matrix * self }
}

// f64 * Matrix
impl Mul<Matrix> for f64 {
    type Output = Matrix;
    fn mul(self, matrix: Matrix) -> Matrix { &matrix * self }
}
