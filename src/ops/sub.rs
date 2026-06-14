use std::ops::Sub;
use crate::matrix::Matrix;

impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Matrix {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Cannot subtract: shape ({}×{}) != ({}×{})",
            self.rows, self.cols, rhs.rows, rhs.cols
        );
        let data = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Matrix::new(self.rows, self.cols, data)
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
