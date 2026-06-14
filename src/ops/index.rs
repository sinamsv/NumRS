use std::ops::{Index, IndexMut};
use crate::matrix::Matrix;

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        assert!(
            row < self.rows && col < self.cols,
            "Index out of bounds: ({}, {}) on Matrix ({}×{})",
            row, col, self.rows, self.cols
        );
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f64 {
        assert!(
            row < self.rows && col < self.cols,
            "Index out of bounds: ({}, {}) on Matrix ({}×{})",
            row, col, self.rows, self.cols
        );
        &mut self.data[row * self.cols + col]
    }
}
