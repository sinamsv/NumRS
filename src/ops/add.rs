use std::ops::Add;
use crate::matrix::Matrix;

// Primary implementation — ref + ref, no ownership consumed
impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Cannot add: shape ({}×{}) != ({}×{})",
            self.rows, self.cols, rhs.rows, rhs.cols
        );
        let data = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Matrix::new(self.rows, self.cols, data)
    }
}

// Owned versions delegate to the ref implementation above
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
