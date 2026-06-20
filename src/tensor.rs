use std::fmt;
use std::ops::{Add, Sub, Mul, Index, IndexMut};
use colored::Colorize;
use num_traits::{Zero, One};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::distributions::Standard;
use rand_distr::{Distribution, StandardNormal};

/// A generic, N-dimensional, row-major (C-order) flat-storage tensor.
///
/// Storage layout mirrors `Matrix`: a single contiguous `Vec<T>` plus a
/// `strides` table so that any rank can be indexed without nested `Vec`s.
/// For a shape `[d0, d1, ..., dn-1]`, `strides[k]` is the product of all
/// dimensions after `k`, so element `[i0, i1, ..., in-1]` lives at
/// `sum(ik * strides[k])` in `data` — exactly the `i * cols + j` scheme
/// `Matrix` uses, generalized to N dimensions.
///
/// `Tensor` is intentionally a separate type from `Matrix` (no shared
/// trait, no breaking changes to the existing 2D API). Conversions
/// between the two can be added later via `From`/`TryFrom`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<T>,
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl<T> Tensor<T> {
    /// Build a tensor from an explicit shape and flat row-major data.
    /// Panics if `data.len()` doesn't match the product of `shape`.
    pub fn from_vec(shape: &[usize], data: Vec<T>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            expected, data.len(),
            "Data length ({}) does not match shape {:?} (expected {})",
            data.len(), shape, expected
        );
        Tensor { shape: shape.to_vec(), strides: compute_strides(shape), data }
    }

    pub fn rank(&self) -> usize { self.shape.len() }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn strides(&self) -> &[usize] { &self.strides }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn as_slice(&self) -> &[T] { &self.data }

    fn flat_index(&self, idx: &[usize]) -> usize {
        assert_eq!(
            idx.len(), self.shape.len(),
            "Index rank {} does not match tensor rank {}", idx.len(), self.shape.len()
        );
        idx.iter().zip(&self.strides).zip(&self.shape)
            .map(|((&i, &s), &dim)| {
                assert!(i < dim, "Index out of bounds: {:?} on shape {:?}", idx, self.shape);
                i * s
            })
            .sum()
    }
}

impl<T: Clone> Tensor<T> {
    /// Fill a tensor of the given shape with a single repeated value.
    pub fn fill(shape: &[usize], value: T) -> Self {
        let size: usize = shape.iter().product();
        Tensor::from_vec(shape, vec![value; size])
    }
}

impl<T: Clone + Zero> Tensor<T> {
    pub fn zeros(shape: &[usize]) -> Self {
        Tensor::fill(shape, T::zero())
    }
}

impl<T: Clone + One> Tensor<T> {
    pub fn ones(shape: &[usize]) -> Self {
        Tensor::fill(shape, T::one())
    }
}

// ── Random constructors (uniform) ────────────────────────────────────────
//
// Works for any `T` that `rand` knows how to sample directly out of thin
// air via the `Standard` distribution — this covers every Rust numeric
// primitive (f32, f64, all integer types), exactly like `Zero`/`One` do
// for `zeros`/`ones`.
impl<T> Tensor<T>
where
    Standard: Distribution<T>,
{
    /// Tensor with elements drawn from a uniform distribution.
    /// For floats this is `[0, 1)`; for integers this is the full range of `T`.
    /// Uses the thread-local RNG — not reproducible across runs.
    /// For reproducible output, use `rand_seeded`.
    pub fn rand(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<T> = (0..size).map(|_| rng.gen::<T>()).collect();
        Tensor::from_vec(shape, data)
    }

    /// Like `rand`, but deterministic: the same `seed` always produces
    /// the same tensor, regardless of machine or run.
    pub fn rand_seeded(shape: &[usize], seed: u64) -> Self {
        let size: usize = shape.iter().product();
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<T> = (0..size).map(|_| rng.gen::<T>()).collect();
        Tensor::from_vec(shape, data)
    }
}

// ── Random constructors (normal) — f32/f64 only ──────────────────────────
//
// A standard normal distribution only makes mathematical sense for
// floating-point types, so unlike `rand`/`zeros`/`ones`, `randn` is NOT
// blanket-generic over `T`. It's implemented individually for `f32` and
// `f64` rather than gated behind `num_traits::Float`, to keep the bound
// simple and avoid pulling Float's full method set into scope here.
macro_rules! impl_randn_for_float {
    ($t:ty) => {
        impl Tensor<$t> {
            /// Tensor with elements drawn from a standard normal
            /// distribution (mean 0, standard deviation 1).
            /// Uses the thread-local RNG — not reproducible across runs.
            /// For reproducible output, use `randn_seeded`.
            pub fn randn(shape: &[usize]) -> Self {
                let size: usize = shape.iter().product();
                let mut rng = rand::thread_rng();
                let data: Vec<$t> = (0..size)
                    .map(|_| rng.sample::<$t, _>(StandardNormal))
                    .collect();
                Tensor::from_vec(shape, data)
            }

            /// Like `randn`, but deterministic: the same `seed` always
            /// produces the same tensor, regardless of machine or run.
            pub fn randn_seeded(shape: &[usize], seed: u64) -> Self {
                let size: usize = shape.iter().product();
                let mut rng = StdRng::seed_from_u64(seed);
                let data: Vec<$t> = (0..size)
                    .map(|_| StandardNormal.sample(&mut rng))
                    .collect();
                Tensor::from_vec(shape, data)
            }
        }
    };
}

impl_randn_for_float!(f32);
impl_randn_for_float!(f64);

// ── Indexing: t[&[i, j, k, ...]] ────────────────────────────────────────
impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;
    fn index(&self, idx: &[usize]) -> &T {
        &self.data[self.flat_index(idx)]
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, idx: &[usize]) -> &mut T {
        let flat = self.flat_index(idx);
        &mut self.data[flat]
    }
}

// ── Element-wise Add / Sub (shape must match exactly) ───────────────────
impl<T: Copy + Add<Output = T>> Add for &Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: &Tensor<T>) -> Tensor<T> {
        assert_eq!(self.shape, rhs.shape, "Cannot add: shape {:?} != {:?}", self.shape, rhs.shape);
        let data = self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a + b).collect();
        Tensor::from_vec(&self.shape, data)
    }
}
impl<T: Copy + Add<Output = T>> Add for Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: Tensor<T>) -> Tensor<T> { &self + &rhs } }
impl<T: Copy + Add<Output = T>> Add<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: &Tensor<T>) -> Tensor<T> { &self + rhs } }
impl<T: Copy + Add<Output = T>> Add<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: Tensor<T>) -> Tensor<T> { self + &rhs } }

impl<T: Copy + Sub<Output = T>> Sub for &Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: &Tensor<T>) -> Tensor<T> {
        assert_eq!(self.shape, rhs.shape, "Cannot subtract: shape {:?} != {:?}", self.shape, rhs.shape);
        let data = self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a - b).collect();
        Tensor::from_vec(&self.shape, data)
    }
}
impl<T: Copy + Sub<Output = T>> Sub for Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: Tensor<T>) -> Tensor<T> { &self - &rhs } }
impl<T: Copy + Sub<Output = T>> Sub<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: &Tensor<T>) -> Tensor<T> { &self - rhs } }
impl<T: Copy + Sub<Output = T>> Sub<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: Tensor<T>) -> Tensor<T> { self - &rhs } }

// ── Scalar multiplication: &tensor * scalar ─────────────────────────────
impl<T: Copy + Mul<Output = T>> Mul<T> for &Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, scalar: T) -> Tensor<T> {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::from_vec(&self.shape, data)
    }
}
impl<T: Copy + Mul<Output = T>> Mul<T> for Tensor<T> { type Output = Tensor<T>; fn mul(self, scalar: T) -> Tensor<T> { &self * scalar } }

impl<T: Copy + Mul<Output = T>> Tensor<T> {
    /// Element-wise (Hadamard) product — shapes must match exactly.
    pub fn hadamard(&self, rhs: &Tensor<T>) -> Tensor<T> {
        assert_eq!(self.shape, rhs.shape, "Hadamard requires equal shapes: {:?} != {:?}", self.shape, rhs.shape);
        let data = self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a * b).collect();
        Tensor::from_vec(&self.shape, data)
    }
}

// ── Display: recursive bracket printer, works for any rank ──────────────
impl<T: fmt::Display + PartialOrd + Zero + Copy> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", format!("Tensor {:?}", self.shape).bold())?;
        fmt_dim(f, &self.data, &self.shape, &self.strides, 0, 0)?;
        writeln!(f)
    }
}

fn fmt_dim<T: fmt::Display + PartialOrd + Zero + Copy>(
    f: &mut fmt::Formatter<'_>,
    data: &[T],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    indent: usize,
) -> fmt::Result {
    if shape.is_empty() {
        return colorize(f, data[offset]);
    }
    if shape.len() == 1 {
        write!(f, "[")?;
        for i in 0..shape[0] {
            if i > 0 { write!(f, ", ")?; }
            colorize(f, data[offset + i * strides[0]])?;
        }
        return write!(f, "]");
    }
    writeln!(f, "[")?;
    for i in 0..shape[0] {
        write!(f, "{}", " ".repeat(indent + 2))?;
        fmt_dim(f, data, &shape[1..], &strides[1..], offset + i * strides[0], indent + 2)?;
        if i < shape[0] - 1 { writeln!(f, ",")?; } else { writeln!(f)?; }
    }
    write!(f, "{}]", " ".repeat(indent))
}

fn colorize<T: fmt::Display + PartialOrd + Zero>(f: &mut fmt::Formatter<'_>, val: T) -> fmt::Result {
    let s = format!("{}", val);
    if val < T::zero() {
        write!(f, "{}", s.yellow())
    } else if val == T::zero() {
        write!(f, "{}", s.dimmed())
    } else {
        write!(f, "{}", s.white())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── rand (generic uniform) ──────────────────────────────────────────
    #[test]
    fn rand_produces_correct_shape_and_len() {
        let t: Tensor<f64> = Tensor::rand(&[2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.len(), 24);
    }

    #[test]
    fn rand_float_values_are_within_unit_interval() {
        let t: Tensor<f64> = Tensor::rand(&[5, 5]);
        for &v in t.as_slice() {
            assert!((0.0..1.0).contains(&v), "value {} out of [0,1)", v);
        }
    }

    #[test]
    fn rand_works_for_integer_types_too() {
        // Just needs to construct successfully with the right length —
        // integer "uniform" spans the full range of the type, so there's
        // no [0,1) bound to check here.
        let t: Tensor<i64> = Tensor::rand(&[3, 3]);
        assert_eq!(t.len(), 9);
    }

    // ── randn (float-only normal) ───────────────────────────────────────
    #[test]
    fn randn_produces_correct_shape_f64() {
        let t: Tensor<f64> = <Tensor<f64>>::randn(&[2, 2, 2]);
        assert_eq!(t.shape(), &[2, 2, 2]);
        assert_eq!(t.len(), 8);
    }

    #[test]
    fn randn_produces_correct_shape_f32() {
        let t: Tensor<f32> = <Tensor<f32>>::randn(&[4]);
        assert_eq!(t.len(), 4);
    }

    #[test]
    fn randn_is_not_trivially_constant() {
        let t: Tensor<f64> = <Tensor<f64>>::randn(&[10, 10]);
        let first = t.as_slice()[0];
        assert!(t.as_slice().iter().any(|&v| v != first));
    }

    // ── seeded reproducibility ────────────────────────────────────────
    #[test]
    fn rand_seeded_is_deterministic() {
        let a: Tensor<f64> = Tensor::rand_seeded(&[3, 3], 42);
        let b: Tensor<f64> = Tensor::rand_seeded(&[3, 3], 42);
        assert_eq!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn rand_seeded_differs_across_seeds() {
        let a: Tensor<f64> = Tensor::rand_seeded(&[3, 3], 1);
        let b: Tensor<f64> = Tensor::rand_seeded(&[3, 3], 2);
        assert_ne!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn randn_seeded_is_deterministic() {
        let a: Tensor<f64> = <Tensor<f64>>::randn_seeded(&[3, 3], 7);
        let b: Tensor<f64> = <Tensor<f64>>::randn_seeded(&[3, 3], 7);
        assert_eq!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn randn_seeded_differs_across_seeds() {
        let a: Tensor<f64> = <Tensor<f64>>::randn_seeded(&[3, 3], 1);
        let b: Tensor<f64> = <Tensor<f64>>::randn_seeded(&[3, 3], 2);
        assert_ne!(a.as_slice(), b.as_slice());
    }
}
