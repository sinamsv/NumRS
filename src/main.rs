mod matrix;
mod ops;

fn main() {
    println!("--- Linear Algebra Library 🦀 ---\n");

    let a = ns_array![[2, -1, 3], [0, 4, 5]];
    let b = ns_array![[1, 0], [-2, 3], [4, -1]];

    println!("A:\n{}", a);
    println!("B:\n{}", b);
    println!("A × B:\n{}", &a * &b);

    let x = ns_array![[3, 1], [2, 4]];
    let y = ns_array![[1, 0], [0, 1]];

    println!("X + Y:\n{}", &x + &y);
    println!("X - Y:\n{}", &x - &y);
    println!("X * 3.0:\n{}", &x * 3.0);
    println!("2.0 * Y:\n{}", 2.0 * &y);

    
    println!("A is still alive:\n{}", a);
}
