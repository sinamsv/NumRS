//! Integration tests — these only use NumRS's *public* API, exactly
//! as an external crate would. They check that multiple operations
//! compose correctly, not just that each one works in isolation.

use numrs::{Matrix, MatrixError, ns_array};

/// Local float-comparison helper (integration tests live outside `src/`,
/// so they can't use the `assert_close!` macro from `test_utils`).
fn close(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

#[test]
fn double_inverse_returns_original_matrix() {
    // (A⁻¹)⁻¹ == A, within floating point tolerance
    let a = ns_array![[3, 1], [2, 4]];
    let inv = a.inverse().unwrap();
    let double_inv = inv.inverse().unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert!(
                close(double_inv[(i, j)], a[(i, j)], 1e-9),
                "mismatch at ({},{}): {} vs {}",
                i, j, double_inv[(i, j)], a[(i, j)]
            );
        }
    }
}

#[test]
fn determinant_of_product_equals_product_of_determinants() {
    // det(A × B) == det(A) × det(B) — classic linear algebra identity
    let a = ns_array![[2, 0], [1, 3]];
    let b = ns_array![[1, 4], [0, 2]];

    let det_a = a.det().unwrap();
    let det_b = b.det().unwrap();
    let det_product = (&a * &b).det().unwrap();

    assert!(
        close(det_product, det_a * det_b, 1e-9),
        "det(A*B)={} but det(A)*det(B)={}",
        det_product, det_a * det_b
    );
}

#[test]
fn transpose_of_product_equals_reversed_product_of_transposes() {
    // (A × B)ᵗ == Bᵗ × Aᵗ
    let a = ns_array![[1, 2], [3, 4], [5, 6]]; // 3×2
    let b = ns_array![[1, 0, 1], [0, 1, 0]];   // 2×3

    let lhs = (&a * &b).transpose();
    let rhs = &b.transpose() * &a.transpose();

    assert_eq!(lhs.data, rhs.data);
    assert_eq!((lhs.rows, lhs.cols), (rhs.rows, rhs.cols));
}

#[test]
fn parallel_and_serial_paths_agree_on_large_addition() {
    // Below PAR_THRESHOLD (1024) add() takes the serial path;
    // above it, the parallel path. Both must produce the same result.
    let n = 50; // 2500 elements — comfortably above threshold
    let data_a: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
    let data_b: Vec<f64> = (0..n * n).map(|i| (i * 2) as f64).collect();

    let a = Matrix::new(n, n, data_a.clone());
    let b = Matrix::new(n, n, data_b.clone());

    let result = (&a + &b).data;
    let expected: Vec<f64> = data_a.iter().zip(data_b.iter())
        .map(|(x, y)| x + y)
        .collect();

    assert_eq!(result, expected);
}

#[test]
fn parallel_and_serial_paths_agree_on_large_multiplication() {
    // Multiply a large matrix by the identity — result must equal
    // the original regardless of whether the parallel row-split
    // path or the serial triple loop was used internally.
    let n = 40; // 1600 elements — above PAR_THRESHOLD
    let data: Vec<f64> = (0..n * n).map(|i| (i % 13) as f64).collect();
    let a = Matrix::new(n, n, data.clone());
    let eye = Matrix::eye(n);

    let result = a.try_mul(&eye).unwrap();
    assert_eq!(result.data, data);
}

#[test]
fn chained_error_propagation_through_question_mark() {
    // Demonstrates that MatrixError works cleanly with `?` across
    // multiple chained fallible operations — important for real
    // user code that composes several NumRS calls.
    fn compute(a: &Matrix, b: &Matrix) -> Result<f64, MatrixError> {
        let sum = a.try_add(b)?;
        let det = sum.det()?;
        Ok(det)
    }

    let a = ns_array![[1, 2], [3, 4]];
    let b = ns_array![[1, 2], [3, 4]];
    let result = compute(&a, &b);
    assert!(result.is_ok());

    // Now force a NonSquare error to propagate through the same chain
    let rect = ns_array![[1, 2, 3]];
    let rect2 = ns_array![[4, 5, 6]];
    match compute(&rect, &rect2) {
        Err(MatrixError::NonSquare { .. }) => {}
        other => panic!("expected NonSquare to propagate, got {:?}", other),
    }
}

#[test]
fn ns_array_macro_produces_matching_matrix_new() {
    // The macro is sugar — it should always agree with the explicit constructor.
    let via_macro = ns_array![[1, 2, 3], [4, 5, 6]];
    let via_new = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    assert_eq!(via_macro.data, via_new.data);
    assert_eq!((via_macro.rows, via_macro.cols), (via_new.rows, via_new.cols));
}
