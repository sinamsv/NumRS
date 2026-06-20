use numrs::{Matrix, ns_array};
use std::time::Instant;

fn main() {
    println!("--- NumRS 0.8.0 🦀 — Parallel Benchmark ---\n");

    let a = ns_array![[1, 2, 3], [4, 5, 6]];
    let b = Matrix::eye(3);
    println!("A × eye(3):\n{}", &a * &b);
    println!("A transposed:\n{}", a.transpose());

    let m = ns_array![[3, 1], [2, 4]];
    println!("det(M) = {:.3}", m.det().unwrap());
    println!("M⁻¹:\n{}", m.inverse().unwrap());

    println!("Random matrix:\n{}", Matrix::rand_seeded(3, 3, 42));
    println!("Random normal matrix:\n{}", Matrix::randn_seeded(3, 3, 42));

    let sizes = [64, 128];

    println!("Matrix Multiplication Benchmark (A × B where A and B are N×N)\n");
    println!("{:>6}  {:>12}  {:>12}  {:>8}",
             "N", "serial (ms)", "parallel (ms)", "speedup");
    println!("{}", "-".repeat(46));

    for &n in &sizes {
        let data_a: Vec<f64> = (0..n*n).map(|i| ((i * 7 + 3) % 97) as f64 / 10.0).collect();
        let data_b: Vec<f64> = (0..n*n).map(|i| ((i * 13 + 5) % 89) as f64 / 10.0).collect();
        let a = Matrix::new(n, n, data_a);
        let b = Matrix::new(n, n, data_b);

        let t0 = Instant::now();
        let mut serial_data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    serial_data[i * n + j] +=
                        a.data[i * n + k] * b.data[k * n + j];
                }
            }
        }
        let serial_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let _ = Matrix::new(n, n, serial_data);

        let t1 = Instant::now();
        let _par = a.try_mul(&b).unwrap();
        let par_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let speedup = serial_ms / par_ms;
        println!("{:>6}  {:>12.2}  {:>13.2}  {:>7.2}×",
                 n, serial_ms, par_ms, speedup);
    }

    println!("\nDone ✓");
}
