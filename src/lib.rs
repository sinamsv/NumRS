pub mod error;
pub mod matrix;
pub mod math;
pub mod ops;
pub mod parallel;

#[cfg(test)]
pub mod test_utils;

pub use matrix::Matrix;
pub use error::MatrixError;
