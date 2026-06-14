use std::ops::Mul;
use crate::matrix::Matrix;

impl Mul for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.cols, rhs.rows,
            "Cannot multiply: left cols ({}) != right rows ({})",
            self.cols, rhs.rows
        );

        let mut data = vec![0.0; self.rows * rhs.cols];

        for i in 0..self.rows {
            for j in 0..rhs.cols {
                for k in 0..self.cols {
                    data[i * rhs.cols + j] +=
                        self.data[i * self.cols + k] * rhs.data[k * rhs.cols + j];
                }
            }
        }

        Matrix::new(self.rows, rhs.cols, data)
    }
}
