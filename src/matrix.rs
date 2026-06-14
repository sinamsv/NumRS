use std::fmt;
use colored::Colorize;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            rows * cols, data.len(),
            "Data length ({}) does not match dimensions ({}×{})",
            data.len(), rows, cols
        );
        Matrix { rows, cols, data }
    }

    pub fn fill(rows: usize, cols: usize, value: f64) -> Self {
        Matrix::new(rows, cols, vec![value; rows * cols])
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
            assert_eq!(rows * cols, total, "Error: rows are not the same size!");
            $crate::matrix::Matrix::new(rows, cols, data)
        }
    };
}
