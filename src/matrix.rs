use std::fmt;
use colored::Colorize;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use crate::error::MatrixError;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Public constructor — panics with a clean `MatrixError` message on bad input.
    /// Internal code should prefer `new_unchecked` when shape is already validated.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        if rows * cols != data.len() {
            panic!(
                "[NumRS] {}",
                MatrixError::InvalidConstruction { rows, cols, data_length: data.len() }
            );
        }
        Matrix { rows, cols, data }
    }

    /// Infallible internal constructor — skips the length check.
    /// Only call this when `rows * cols == data.len()` is already guaranteed.
    pub(crate) fn new_unchecked(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        debug_assert_eq!(
            rows * cols, data.len(),
            "new_unchecked called with mismatched dimensions — this is a NumRS bug"
        );
        Matrix { rows, cols, data }
    }

    pub fn fill(rows: usize, cols: usize, value: f64) -> Self {
        Matrix::new_unchecked(rows, cols, vec![value; rows * cols])
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix::fill(rows, cols, 0.0)
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Matrix::fill(rows, cols, 1.0)
    }

    pub fn eye(n: usize) -> Self {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    // ── Random constructors ─────────────────────────────────────────────

    /// Matrix with elements drawn from a uniform distribution over `[0, 1)`.
    /// Uses the thread-local RNG — not reproducible across runs.
    /// For reproducible output, use `rand_seeded`.
    pub fn rand(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen::<f64>()).collect();
        Matrix::new_unchecked(rows, cols, data)
    }

    /// Matrix with elements drawn from a standard normal distribution
    /// (mean 0, standard deviation 1).
    /// Uses the thread-local RNG — not reproducible across runs.
    /// For reproducible output, use `randn_seeded`.
    pub fn randn(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        Matrix::new_unchecked(rows, cols, data)
    }

    /// Like `rand`, but deterministic: the same `seed` always produces
    /// the same matrix, regardless of machine or run.
    pub fn rand_seeded(rows: usize, cols: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f64> = (0..rows * cols).map(|_| rng.gen::<f64>()).collect();
        Matrix::new_unchecked(rows, cols, data)
    }

    /// Like `randn`, but deterministic: the same `seed` always produces
    /// the same matrix, regardless of machine or run.
    pub fn randn_seeded(rows: usize, cols: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        Matrix::new_unchecked(rows, cols, data)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let formatted: Vec<String> = self.data.iter()
            .map(|&x| format!("{:.3}", x))
            .collect();

        let col_widths: Vec<usize> = (0..self.cols)
            .map(|j| {
                (0..self.rows)
                    .map(|i| formatted[i * self.cols + j].len())
                    .max()
                    .unwrap_or(0)
            })
            .collect();

        writeln!(f, "{}", format!("Matrix ({}×{})", self.rows, self.cols).bold())?;

        for i in 0..self.rows {
            let (left, right) = match (self.rows, i) {
                (1, _)                        => ("[ ", " ]"),
                (_, x) if x == 0             => ("┌ ", " ┐"),
                (_, x) if x == self.rows - 1 => ("└ ", " ┘"),
                _                            => ("│ ", " │"),
            };

            write!(f, "{}", left.cyan().bold())?;

            for j in 0..self.cols {
                let val = self.data[i * self.cols + j];
                let s = format!("{:>width$.3}", val, width = col_widths[j]);

                let colored_s = if val < 0.0 {
                    s.yellow()
                } else if val == 0.0 {
                    s.dimmed()
                } else {
                    s.white()
                };

                write!(f, "{}", colored_s)?;
                if j < self.cols - 1 { write!(f, "  ")?; }
            }

            writeln!(f, "{}", right.cyan().bold())?;
        }

        Ok(())
    }
}

#[macro_export]
macro_rules! ns_array {
    ( $( [ $( $x:expr ),* ] ),* ) => {
        {
            let mut rows = 0;
            let mut data = Vec::new();
            $(
                rows += 1;
                $( data.push($x as f64); )*
            )*
            let total = data.len();
            let cols = if rows > 0 { total / rows } else { 0 };
            assert_eq!(rows * cols, total, "[NumRS] ns_array!: rows are not the same length");
            $crate::matrix::Matrix::new(rows, cols, data)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── rand ────────────────────────────────────────────────────────────
    #[test]
    fn rand_produces_correct_shape() {
        let m = Matrix::rand(4, 5);
        assert_eq!((m.rows, m.cols), (4, 5));
        assert_eq!(m.data.len(), 20);
    }

    #[test]
    fn rand_values_are_within_unit_interval() {
        let m = Matrix::rand(10, 10);
        for &v in &m.data {
            assert!((0.0..1.0).contains(&v), "value {} out of [0,1)", v);
        }
    }

    // ── randn ───────────────────────────────────────────────────────────
    #[test]
    fn randn_produces_correct_shape() {
        let m = Matrix::randn(3, 7);
        assert_eq!((m.rows, m.cols), (3, 7));
        assert_eq!(m.data.len(), 21);
    }

    #[test]
    fn randn_is_not_trivially_constant() {
        // Sanity check: a real normal sample over enough elements
        // should not collapse to a single repeated value.
        let m = Matrix::randn(20, 20);
        let first = m.data[0];
        assert!(m.data.iter().any(|&v| v != first));
    }

    // ── seeded reproducibility ────────────────────────────────────────
    #[test]
    fn rand_seeded_is_deterministic() {
        let a = Matrix::rand_seeded(5, 5, 42);
        let b = Matrix::rand_seeded(5, 5, 42);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn rand_seeded_differs_across_seeds() {
        let a = Matrix::rand_seeded(5, 5, 1);
        let b = Matrix::rand_seeded(5, 5, 2);
        assert_ne!(a.data, b.data);
    }

    #[test]
    fn randn_seeded_is_deterministic() {
        let a = Matrix::randn_seeded(5, 5, 7);
        let b = Matrix::randn_seeded(5, 5, 7);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn randn_seeded_differs_across_seeds() {
        let a = Matrix::randn_seeded(5, 5, 1);
        let b = Matrix::randn_seeded(5, 5, 2);
        assert_ne!(a.data, b.data);
    }
}
