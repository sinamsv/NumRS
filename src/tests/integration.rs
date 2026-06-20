use numrs::{Matrix, MatrixError, ns_array};

fn close(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

#[test]
fn double_inverse_returns_original_matrix() {
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
    let a = ns_array![[1, 2], [3, 4], [5, 6]];
    let b = ns_array![[1, 0, 1], [0, 1, 0]];

    let lhs = (&a * &b).transpose();
    let rhs = &b.transpose() * &a.transpose();

    assert_eq!(lhs.data, rhs.data);
    assert_eq!((lhs.rows, lhs.cols), (rhs.rows, rhs.cols));
}

#[test]
fn parallel_and_serial_paths_agree_on_large_addition() {
    let n = 50;
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
    let n = 40;
    let data: Vec<f64> = (0..n * n).map(|i| (i % 13) as f64).collect();
    let a = Matrix::new(n, n, data.clone());
    let eye = Matrix::eye(n);

    let result = a.try_mul(&eye).unwrap();
    assert_eq!(result.data, data);
}

#[test]
fn chained_error_propagation_through_question_mark() {
    fn compute(a: &Matrix, b: &Matrix) -> Result<f64, MatrixError> {
        let sum = a.try_add(b)?;
        let det = sum.det()?;
        Ok(det)
    }

    let a = ns_array![[1, 2], [3, 4]];
    let b = ns_array![[1, 2], [3, 4]];
    let result = compute(&a, &b);
    assert!(result.is_ok());

    let rect = ns_array![[1, 2, 3]];
    let rect2 = ns_array![[4, 5, 6]];
    match compute(&rect, &rect2) {
        Err(MatrixError::NonSquare { .. }) => {}
        other => panic!("expected NonSquare to propagate, got {:?}", other),
    }
}

#[test]
fn ns_array_macro_produces_matching_matrix_new() {
    let via_macro = ns_array![[1, 2, 3], [4, 5, 6]];
    let via_new = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    assert_eq!(via_macro.data, via_new.data);
    assert_eq!((via_macro.rows, via_macro.cols), (via_new.rows, via_new.cols));
}

// ── New: rand/randn integration tests ───────────────────────────────────

#[test]
fn matrix_rand_seeded_is_reproducible_across_calls() {
    let a = Matrix::rand_seeded(6, 6, 123);
    let b = Matrix::rand_seeded(6, 6, 123);
    assert_eq!(a.data, b.data);
}

#[test]
fn matrix_randn_then_used_in_matmul_does_not_panic() {
    // Sanity check that random matrices compose with the rest of the
    // public API exactly like any other matrix would.
    let a = Matrix::randn_seeded(4, 4, 99);
    let b = Matrix::eye(4);
    let product = &a * &b;
    assert_eq!(product.data, a.data);
}

#[test]
fn tensor_rand_seeded_is_reproducible_across_calls() {
    let a: numrs::Tensor<f64> = numrs::Tensor::rand_seeded(&[3, 3], 55);
    let b: numrs::Tensor<f64> = numrs::Tensor::rand_seeded(&[3, 3], 55);
    assert_eq!(a.as_slice(), b.as_slice());
}

#[test]
fn tensor_randn_seeded_is_reproducible_across_calls() {
    let a: numrs::Tensor<f64> = <numrs::Tensor<f64>>::randn_seeded(&[2, 2, 2], 8);
    let b: numrs::Tensor<f64> = <numrs::Tensor<f64>>::randn_seeded(&[2, 2, 2], 8);
    assert_eq!(a.as_slice(), b.as_slice());
}
