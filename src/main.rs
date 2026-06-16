use numrs::{Matrix, MatrixError, ns_array};

fn main() {
    println!("--- NumRS 0.6.1 🦀 — Error Handling Demo ---\n");

    // ── Operators still work cleanly ───────────────────────────────────────
    let a = ns_array![[1, 2, 3], [4, 5, 6]];
    let b = Matrix::eye(3);
    println!("A:\n{}", a);
    println!("A × eye(3):\n{}", &a * &b);
    println!("A transposed:\n{}", a.transpose());

    // ── det() now returns Result ───────────────────────────────────────────
    let m = ns_array![[3, 1], [2, 4]];
    match m.det() {
        Ok(d)  => println!("det(M) = {:.3}\n", d),
        Err(e) => println!("det error: {}\n", e),
    }

    // ── inverse() returns Result instead of Option ─────────────────────────
    match m.inverse() {
        Ok(inv) => println!("M⁻¹:\n{}", inv),
        Err(e)  => println!("inverse error: {}\n", e),
    }

    // ── NonInvertible error ────────────────────────────────────────────────
    let singular = ns_array![[1, 2], [2, 4]];
    match singular.inverse() {
        Ok(_)  => println!("Got inverse (unexpected)"),
        Err(e) => println!("Singular matrix → Err: {}\n", e),
    }

    // ── NonSquare error ────────────────────────────────────────────────────
    let rect = ns_array![[1, 2, 3], [4, 5, 6]];
    match rect.det() {
        Ok(d)  => println!("det = {}", d),
        Err(e) => println!("Non-square → Err: {}\n", e),
    }

    // ── try_add — safe addition ────────────────────────────────────────────
    let x = ns_array![[1, 2], [3, 4]];
    let y = ns_array![[5, 6], [7, 8]];
    match x.try_add(&y) {
        Ok(sum) => println!("try_add:\n{}", sum),
        Err(e)  => println!("try_add error: {}\n", e),
    }

    // ── try_add with mismatched shapes ────────────────────────────────────
    let z = ns_array![[1, 2, 3]];
    match x.try_add(&z) {
        Ok(_)  => println!("Added (unexpected)"),
        Err(e) => println!("Shape mismatch → Err: {}\n", e),
    }

    // ── try_mul — safe multiplication ─────────────────────────────────────
    let p = ns_array![[1, 2], [3, 4]];
    let q = ns_array![[1, 0], [0, 1]];
    match p.try_mul(&q) {
        Ok(prod) => println!("try_mul (P × I):\n{}", prod),
        Err(e)   => println!("try_mul error: {}\n", e),
    }

    // ── Matching on error variant ──────────────────────────────────────────
    let bad = ns_array![[1, 2, 3]];
    if let Err(MatrixError::NonSquare { rows, cols }) = bad.det() {
        println!("Caught NonSquare: {}×{} is not a square matrix\n", rows, cols);
    }

    println!("Done ✓");
}

