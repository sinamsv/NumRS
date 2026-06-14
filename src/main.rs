mod matrix;
mod math;
mod ops;

use matrix::Matrix;

fn main() {
    println!("--- Stage 4: Math Operations 🦀 ---\n");

    let a = ns_array![
        [1, 2, 3],
        [4, 5, 6]
    ];

    // ── Transpose ─────────────────────────────────────────
    println!("A:\n{}", a);
    println!("A transposed:\n{}", a.transpose());

    // ── Hadamard ──────────────────────────────────────────
    let b = ns_array![
        [2, 0, 1],
        [1, 3, 2]
    ];
    println!("B:\n{}", b);
    println!("A ⊙ B (Hadamard):\n{}", a.hadamard(&b));

    // ── Norm ──────────────────────────────────────────────
    println!("Frobenius norm of A: {:.4}\n", a.norm());

    // ── Determinant ───────────────────────────────────────
    let sq = ns_array![
        [3, 8],
        [4, 6]
    ];
    println!("Square matrix:\n{}", sq);
    println!("det = {:.4}\n", sq.det());

    // ── Inverse ───────────────────────────────────────────
    let m = ns_array![
        [1, 2],
        [3, 4]
    ];
    println!("M:\n{}", m);

    match m.inverse() {
        Some(inv) => {
            println!("M⁻¹:\n{}", inv);
            // Sanity check: M * M⁻¹ should be identity
            println!("M × M⁻¹ (should be identity):\n{}", &m * &inv);
        }
        None => println!("Matrix is singular — no inverse exists.\n"),
    }

    // ── Singular matrix test ───────────────────────────────
    let singular = ns_array![
        [1, 2],
        [2, 4]
    ];
    println!("Singular matrix:\n{}", singular);
    println!("det = {:.4}", singular.det());
    match singular.inverse() {
        Some(_) => println!("Has inverse (unexpected!)"),
        None    => println!("Correctly detected: no inverse exists.\n"),
    }
}
