pub mod error;
pub mod matrix;
pub mod math;
pub mod ops;
pub mod parallel;
pub mod tensor;

#[cfg(test)]
pub mod test_utils;

pub use matrix::Matrix;
pub use error::{MatrixError, TensorError};
pub use tensor::Tensor;
