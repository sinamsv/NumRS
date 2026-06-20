# NumRS Examples

This guide provides full, runnable examples of how to use NumRS for common linear algebra tasks.

---

## 1. Solving Linear Systems ($Ax = b$)

To solve a system of linear equations like:
$3x + y = 10$
$2x + 4y = 20$

We can represent this as $Ax = b$ and solve for $x$ using $x = A^{-1}b$.

```rust
use numrs::{ns_array, Matrix};

fn main() {
    let a = ns_array![[3.0, 1.0], [2.0, 4.0]];
    let b = ns_array![[10.0], [20.0]];

    // Solve x = A⁻¹ * b
    let a_inv = a.inverse().expect("Matrix must be invertible");
    let x = &a_inv * &b;

    println!("Solution x:\n{}", x);
    // Expected: x ≈ 2, y ≈ 4
}
```

---

## 2. Markov Chain Simulation

Simulate a simple 2-state Markov chain (e.g., Weather: Sunny/Rainy).

```rust
use numrs::{ns_array, Matrix};

fn main() {
    // Transition matrix:
    //         Sunny  Rainy
    // Sunny [ 0.9,   0.1 ]
    // Rainy [ 0.5,   0.5 ]
    let p = ns_array![[0.9, 0.1], [0.5, 0.5]];

    // Initial state: 100% Sunny
    let mut state = ns_array![[1.0, 0.0]];

    println!("Initial state: {}", state);

    for i in 1..=5 {
        state = &state * &p;
        println!("After step {}: {}", i, state);
    }
}
```

---

## 3. Least Squares Fitting

Find the best-fit line $y = mx + c$ for a set of points $(1, 2), (2, 3), (3, 5), (4, 4)$.
The normal equation is $x = (A^T A)^{-1} A^T b$.

```rust
use numrs::{ns_array, Matrix};

fn main() {
    // Design matrix A (ones in second column for intercept c)
    let a = ns_array![
        [1.0, 1.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [4.0, 1.0]
    ];

    // Observation vector b
    let b = ns_array![[2.0], [3.0], [5.0], [4.0]];

    let at = a.transpose();
    let at_a_inv = (&at * &a).inverse().expect("Matrix not invertible");
    let weights = &(&at_a_inv * &at) * &b;

    println!("Weights (m, c):\n{}", weights);
}
```

---

## 4. Graph Path Counting

Given an adjacency matrix $A$ of a graph, $(A^k)_{ij}$ gives the number of paths of length $k$ from node $i$ to node $j$.

```rust
use numrs::{ns_array, Matrix};

fn main() {
    // A directed graph: 0 -> 1, 1 -> 2, 2 -> 0, 2 -> 1
    let a = ns_array![
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ];

    // Paths of length 3
    let a2 = &a * &a;
    let a3 = &a2 * &a;

    println!("Number of paths of length 3:\n{}", a3);
}
```

---

## 5. 2D Transformations

Applying rotation and scaling to a 2D point.

```rust
use numrs::{ns_array, Matrix};

fn main() {
    let point = ns_array![[1.0], [0.0]]; // Point (1, 0)

    // 90-degree rotation matrix
    let angle = std::f64::consts::FRAC_PI_2;
    let rotation = ns_array![
        [angle.cos(), -angle.sin()],
        [angle.sin(),  angle.cos()]
    ];

    // Scaling matrix (2x)
    let scale = ns_array![
        [2.0, 0.0],
        [0.0, 2.0]
    ];

    let transformed = &scale * &(&rotation * &point);
    println!("Transformed point:\n{}", transformed);
}
```

---

## 6. Hill Cipher (Simplified)

Encryption using a key matrix. Note: For traditional Hill Cipher, operations are performed modulo 26. Since NumRS uses `f64`, this example demonstrates the linear algebra core.

```rust
use numrs::{ns_array, Matrix};

fn main() {
    // Key matrix (must be invertible)
    let key = ns_array![[6.0, 24.0, 1.0], [13.0, 16.0, 10.0], [20.0, 17.0, 15.0]];

    // Plaintext "ACT" -> [0, 2, 19]
    let plaintext = ns_array![[0.0], [2.0], [19.0]];

    // Encrypt: C = K * P
    let ciphertext = &key * &plaintext;
    println!("Raw ciphertext:\n{}", ciphertext);

    // Decrypt: P = K⁻¹ * C
    let key_inv = key.inverse().expect("Key must be invertible");
    let decrypted = &key_inv * &ciphertext;
    println!("Decrypted plaintext:\n{}", decrypted);
}
```
