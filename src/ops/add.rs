use std::ops::Add;
use rayon::prelude::*;
use crate::matrix::Matrix;
use crate::error::MatrixError;
use crate::parallel::should_parallelize;

fn check_same_shape(lhs: &Matrix, rhs: &Matrix) -> Result<(), MatrixError> {
    if lhs.rows == rhs.rows && lhs.cols == rhs.cols {
        Ok(())
    } else {
        Err(MatrixError::ShapeMismatch {
            expected: (lhs.rows, lhs.cols),
            found:    (rhs.rows, rhs.cols),
        })
    }
}

impl Matrix {
    pub fn try_add(&self, rhs: &Matrix) -> Result<Matrix, MatrixError> {
        check_same_shape(self, rhs)?;
        let data = if should_parallelize(self.data.len()) {
            self.data.par_iter()
                .zip(rhs.data.par_iter())
                .map(|(a, b)| a + b)
                .collect()
        } else {
            self.data.iter()
                .zip(rhs.data.iter())
                .map(|(a, b)| a + b)
                .collect()
        };
        Ok(Matrix::new_unchecked(self.rows, self.cols, data))
    }
}

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
        match check_same_shape(self, rhs) {
            Ok(_)  => self.try_add(rhs).unwrap(),
            Err(e) => panic!("[NumRS] Add failed: {}", e),
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ns_array;
    use crate::assert_close;

    #[test]
    fn add_produces_correct_values() {
        let a = ns_array![[1, 2], [3, 4]];
        let b = ns_array![[5, 6], [7, 8]];
        let c = &a + &b;
        assert_eq!(c.data, vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!((c.rows, c.cols), (2, 2));
    }

    #[test]
    fn add_works_for_all_four_ownership_combinations() {
        let a = ns_array![[1, 2]];
        let b = ns_array![[3, 4]];

        let r1 = &a + &b;
        let r2 = a.clone() + b.clone();
        let r3 = a.clone() + &b;
        let r4 = &a + b.clone();

        for r in [&r1, &r2, &r3, &r4] {
            assert_eq!(r.data, vec![4.0, 6.0]);
        }
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn add_panics_on_shape_mismatch() {
        let a = ns_array![[1, 2], [3, 4]];
        let b = ns_array![[1, 2, 3]];
        let _ = &a + &b;
    }

    #[test]
    fn try_add_returns_ok_on_matching_shapes() {
        let a = ns_array![[1, 2], [3, 4]];
        let b = ns_array![[1, 1], [1, 1]];
        let result = a.try_add(&b);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().data, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn try_add_returns_shape_mismatch_error() {
        let a = ns_array![[1, 2], [3, 4]];
        let b = ns_array![[1, 2, 3]];

        match a.try_add(&b) {
            Err(MatrixError::ShapeMismatch { expected, found }) => {
                assert_eq!(expected, (2, 2));
                assert_eq!(found, (1, 3));
            }
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn try_add_matches_serial_and_parallel_paths() {
        let n = 40;
        let data: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let a = Matrix::new(n, n, data.clone());
        let b = Matrix::new(n, n, data);

        let result = a.try_add(&b).unwrap();
        for (i, &v) in result.data.iter().enumerate() {
            assert_close!(v, (i as f64) * 2.0);
        }
    }
}
