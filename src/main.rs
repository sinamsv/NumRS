use numrs::{Matrix, ns_array};

fn main() {
    println!("--- NumRS 🦀 ---\n");

    let a = ns_array![
        [1, 2, 3],
        [4, 5, 6]
    ];

    let b = Matrix::eye(3);

    println!("A:\n{}", a);
    println!("eye(3):\n{}", b);
    println!("A × eye:\n{}", &a * &b);
    println!("A transposed:\n{}", a.transpose());

    let m = ns_array![[3, 1], [2, 4]];
    println!("det(M) = {:.3}", m.det());

    match m.inverse() {
        Some(inv) => println!("M⁻¹:\n{}", inv),
        None      => println!("Singular — no inverse."),
    }
}

