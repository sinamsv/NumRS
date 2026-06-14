use std::ops::Mul;
use crate::matrix::Matrix;

// &Matrix * f64
impl Mul<f64> for &Matrix {
    type Output = Matrix;
    fn mul(self, scalar: f64) -> Matrix {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Matrix::new(self.rows, self.cols, data)
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
