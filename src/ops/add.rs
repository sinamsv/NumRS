use std::ops::Add;
use crate::matrix::Matrix;

impl Add for Matrix {
    type Output = Matrix;

    fn add(self, rhs: Self) -> Self::Output {
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
