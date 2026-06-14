mod matrix;
mod ops;

fn main() {
    println!("--- Linear Algebra Library 🦀 ---\n");

    let a = ns_array![
        [2, -1, 3],
        [0,  4, 5]
    ];
    let b = ns_array![
        [1,  0],
        [-2, 3],
        [4, -1]
    ];    
    println!("A:\n{}", a);
    println!("B:\n{}", b);
    println!("A × B × C:\n{}", a.clone() * b);

    let x = ns_array![[1, 2, 3], [4, 5, 6]];
    let y = ns_array![[6, 5, 4], [3, 2, 1]];
    println!("X + Y:\n{}", x + y);
}
