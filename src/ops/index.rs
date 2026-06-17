use std::ops::{Index, IndexMut};
use crate::matrix::Matrix;
use crate::error::MatrixError;

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        if row >= self.rows || col >= self.cols {
            panic!(
                "[NumRS] {}",
                MatrixError::IndexOutOfBounds {
                    index:      (row, col),
                    dimensions: (self.rows, self.cols),
                }
            );
        }
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f64 {
        if row >= self.rows || col >= self.cols {
            panic!(
                "[NumRS] {}",
                MatrixError::IndexOutOfBounds {
                    index:      (row, col),
                    dimensions: (self.rows, self.cols),
                }
            );
        }
        &mut self.data[row * self.cols + col]
    }
}

#[cfg(test)]
mod tests {
    use crate::ns_array;

    #[test]
    fn index_reads_correct_value() {
        let m = ns_array![[1, 2], [3, 4]];
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn index_mut_writes_correct_value() {
        let mut m = ns_array![[1, 2], [3, 4]];
        m[(0, 1)] = 99.0;
        assert_eq!(m[(0, 1)], 99.0);
        // Neighboring elements must stay untouched
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_panics_on_row_out_of_bounds() {
        let m = ns_array![[1, 2], [3, 4]]; // 2×2
        let _ = m[(2, 0)]; // row 2 doesn't exist
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_panics_on_col_out_of_bounds() {
        let m = ns_array![[1, 2], [3, 4]]; // 2×2
        let _ = m[(0, 5)]; // col 5 doesn't exist
    }
}
