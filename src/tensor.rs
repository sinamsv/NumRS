use std::fmt;
use std::ops::{Add, Sub, Mul, Index, IndexMut};
use colored::Colorize;
use num_traits::{Zero, One};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::distributions::Standard;
use rand_distr::{Distribution, StandardNormal};
use crate::error::TensorError;

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

fn check_same_shape(lhs: &[usize], rhs: &[usize]) -> Result<(), TensorError> {
    if lhs == rhs {
        Ok(())
    } else {
        Err(TensorError::ShapeMismatch {
            expected: lhs.to_vec(),
            found:    rhs.to_vec(),
        })
    }
}

impl<T> Tensor<T> {
    /// Fallible constructor — build a tensor from an explicit shape and
    /// flat row-major data. Returns `Err(TensorError::InvalidConstruction)`
    /// if `data.len()` doesn't match the product of `shape`.
    pub fn try_from_vec(shape: &[usize], data: Vec<T>) -> Result<Self, TensorError> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(TensorError::InvalidConstruction {
                shape:        shape.to_vec(),
                expected_len: expected,
                found_len:    data.len(),
            });
        }
        Ok(Tensor { shape: shape.to_vec(), strides: compute_strides(shape), data })
    }

    /// Build a tensor from an explicit shape and flat row-major data.
    /// Panics with a `TensorError::InvalidConstruction` message if
    /// `data.len()` doesn't match the product of `shape`.
    /// For a non-panicking alternative, use `try_from_vec`.
    pub fn from_vec(shape: &[usize], data: Vec<T>) -> Self {
        match Tensor::try_from_vec(shape, data) {
            Ok(t)  => t,
            Err(e) => panic!("[NumRS] {}", e),
        }
    }

    /// Infallible internal constructor — skips the length check.
    /// Only call this when `shape.iter().product() == data.len()` is
    /// already guaranteed.
    pub(crate) fn from_vec_unchecked(shape: &[usize], data: Vec<T>) -> Self {
        debug_assert_eq!(
            shape.iter().product::<usize>(), data.len(),
            "from_vec_unchecked called with mismatched shape — this is a NumRS bug"
        );
        Tensor { shape: shape.to_vec(), strides: compute_strides(shape), data }
    }

    pub fn rank(&self) -> usize { self.shape.len() }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn strides(&self) -> &[usize] { &self.strides }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn as_slice(&self) -> &[T] { &self.data }

    /// Fallible flat-index resolution — checks rank and per-axis bounds
    /// before computing the offset into `data`.
    fn try_flat_index(&self, idx: &[usize]) -> Result<usize, TensorError> {
        if idx.len() != self.shape.len() {
            return Err(TensorError::RankMismatch {
                expected_rank: self.shape.len(),
                found_rank:    idx.len(),
            });
        }
        for (&i, &dim) in idx.iter().zip(&self.shape) {
            if i >= dim {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx.to_vec(),
                    shape: self.shape.clone(),
                });
            }
        }
        Ok(idx.iter().zip(&self.strides).map(|(&i, &s)| i * s).sum())
    }

    /// Fallible element access — `Result` counterpart to `Index`/`IndexMut`.
    pub fn try_get(&self, idx: &[usize]) -> Result<&T, TensorError> {
        let flat = self.try_flat_index(idx)?;
        Ok(&self.data[flat])
    }

    /// Fallible mutable element access — `Result` counterpart to `IndexMut`.
    pub fn try_get_mut(&mut self, idx: &[usize]) -> Result<&mut T, TensorError> {
        let flat = self.try_flat_index(idx)?;
        Ok(&mut self.data[flat])
    }
}

impl<T: Clone> Tensor<T> {
    /// Fill a tensor of the given shape with a single repeated value.
    pub fn fill(shape: &[usize], value: T) -> Self {
        let size: usize = shape.iter().product();
        Tensor::from_vec_unchecked(shape, vec![value; size])
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
        Tensor::from_vec_unchecked(shape, data)
    }

    /// Like `rand`, but deterministic: the same `seed` always produces
    /// the same tensor, regardless of machine or run.
    pub fn rand_seeded(shape: &[usize], seed: u64) -> Self {
        let size: usize = shape.iter().product();
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<T> = (0..size).map(|_| rng.gen::<T>()).collect();
        Tensor::from_vec_unchecked(shape, data)
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
                Tensor::from_vec_unchecked(shape, data)
            }

            /// Like `randn`, but deterministic: the same `seed` always
            /// produces the same tensor, regardless of machine or run.
            pub fn randn_seeded(shape: &[usize], seed: u64) -> Self {
                let size: usize = shape.iter().product();
                let mut rng = StdRng::seed_from_u64(seed);
                let data: Vec<$t> = (0..size)
                    .map(|_| StandardNormal.sample(&mut rng))
                    .collect();
                Tensor::from_vec_unchecked(shape, data)
            }
        }
    };
}

impl_randn_for_float!(f32);
impl_randn_for_float!(f64);

// ── Indexing: t[&[i, j, k, ...]] ────────────────────────────────────────
//
// `Index`/`IndexMut` can't return `Result` (the trait signature requires
// `&Output`/`&mut Output`), so they route through `try_get`/`try_get_mut`
// and unwrap — panicking with the same clean `TensorError::Display`
// message that `Matrix`'s `Index` impl produces via `MatrixError`.
// Prefer `try_get`/`try_get_mut` directly in contexts where you want a
// `Result` instead of a panic.
impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;
    fn index(&self, idx: &[usize]) -> &T {
        match self.try_get(idx) {
            Ok(v)  => v,
            Err(e) => panic!("[NumRS] {}", e),
        }
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, idx: &[usize]) -> &mut T {
        match self.try_flat_index(idx) {
            Ok(flat) => &mut self.data[flat],
            Err(e)   => panic!("[NumRS] {}", e),
        }
    }
}

// ── Element-wise Add / Sub (shape must match exactly) ───────────────────
//
// Safe variants, returning `Result` — the `Add`/`Sub` operator impls below
// delegate to these and unwrap, mirroring `Matrix::try_add` / `try_sub`.
impl<T: Copy + Add<Output = T>> Tensor<T> {
    pub fn try_add(&self, rhs: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        check_same_shape(&self.shape, &rhs.shape)?;
        let data = self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a + b).collect();
        Ok(Tensor::from_vec_unchecked(&self.shape, data))
    }
}

impl<T: Copy + Sub<Output = T>> Tensor<T> {
    pub fn try_sub(&self, rhs: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        check_same_shape(&self.shape, &rhs.shape)?;
        let data = self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a - b).collect();
        Ok(Tensor::from_vec_unchecked(&self.shape, data))
    }
}

impl<T: Copy + Add<Output = T>> Add for &Tensor<T> {
    type Output = Tensor<T>;
    fn add(self, rhs: &Tensor<T>) -> Tensor<T> {
        match self.try_add(rhs) {
            Ok(t)  => t,
            Err(e) => panic!("[NumRS] Add failed: {}", e),
        }
    }
}
impl<T: Copy + Add<Output = T>> Add for Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: Tensor<T>) -> Tensor<T> { &self + &rhs } }
impl<T: Copy + Add<Output = T>> Add<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: &Tensor<T>) -> Tensor<T> { &self + rhs } }
impl<T: Copy + Add<Output = T>> Add<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn add(self, rhs: Tensor<T>) -> Tensor<T> { self + &rhs } }

impl<T: Copy + Sub<Output = T>> Sub for &Tensor<T> {
    type Output = Tensor<T>;
    fn sub(self, rhs: &Tensor<T>) -> Tensor<T> {
        match self.try_sub(rhs) {
            Ok(t)  => t,
            Err(e) => panic!("[NumRS] Sub failed: {}", e),
        }
    }
}
impl<T: Copy + Sub<Output = T>> Sub for Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: Tensor<T>) -> Tensor<T> { &self - &rhs } }
impl<T: Copy + Sub<Output = T>> Sub<&Tensor<T>> for Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: &Tensor<T>) -> Tensor<T> { &self - rhs } }
impl<T: Copy + Sub<Output = T>> Sub<Tensor<T>> for &Tensor<T> { type Output = Tensor<T>; fn sub(self, rhs: Tensor<T>) -> Tensor<T> { self - &rhs } }

// ── Scalar multiplication: &tensor * scalar ─────────────────────────────
//
// Scalar multiplication can never fail on shape grounds (every element is
// just multiplied independently), so there's no `try_mul_scalar` — nothing
// here can panic on a value this library controls.
impl<T: Copy + Mul<Output = T>> Mul<T> for &Tensor<T> {
    type Output = Tensor<T>;
    fn mul(self, scalar: T) -> Tensor<T> {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::from_vec_unchecked(&self.shape, data)
    }
}
impl<T: Copy + Mul<Output = T>> Mul<T> for Tensor<T> { type Output = Tensor<T>; fn mul(self, scalar: T) -> Tensor<T> { &self * scalar } }

impl<T: Copy + Mul<Output = T>> Tensor<T> {
    /// Element-wise (Hadamard) product — fallible; shapes must match exactly.
    pub fn try_hadamard(&self, rhs: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        check_same_shape(&self.shape, &rhs.shape)?;
        let data = self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a * b).collect();
        Ok(Tensor::from_vec_unchecked(&self.shape, data))
    }

    /// Element-wise (Hadamard) product — panics with a clean
    /// `TensorError::Display` message on shape mismatch.
    /// For a non-panicking alternative, use `try_hadamard`.
    pub fn hadamard(&self, rhs: &Tensor<T>) -> Tensor<T> {
        match self.try_hadamard(rhs) {
            Ok(t)  => t,
            Err(e) => panic!("[NumRS] Hadamard failed: {}", e),
        }
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
    use crate::error::TensorError;

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

    // ── try_from_vec / from_vec error handling ──────────────────────────
    #[test]
    fn try_from_vec_returns_ok_on_matching_length() {
        let t = Tensor::try_from_vec(&[2, 2], vec![1, 2, 3, 4]);
        assert!(t.is_ok());
    }

    #[test]
    fn try_from_vec_returns_invalid_construction_error() {
        match Tensor::try_from_vec(&[2, 2], vec![1, 2, 3]) {
            Err(TensorError::InvalidConstruction { shape, expected_len, found_len }) => {
                assert_eq!(shape, vec![2, 2]);
                assert_eq!(expected_len, 4);
                assert_eq!(found_len, 3);
            }
            other => panic!("expected InvalidConstruction, got {:?}", other),
        }
    }

    #[test]
    #[should_panic(expected = "invalid construction")]
    fn from_vec_panics_on_mismatched_length() {
        let _: Tensor<i32> = Tensor::from_vec(&[2, 2], vec![1, 2, 3]);
    }

    // ── try_get / try_get_mut / Index error handling ────────────────────
    #[test]
    fn try_get_reads_correct_value() {
        let t = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
        assert_eq!(*t.try_get(&[0, 1]).unwrap(), 2);
        assert_eq!(*t.try_get(&[1, 0]).unwrap(), 3);
    }

    #[test]
    fn try_get_returns_rank_mismatch_error() {
        let t = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
        match t.try_get(&[0]) {
            Err(TensorError::RankMismatch { expected_rank, found_rank }) => {
                assert_eq!(expected_rank, 2);
                assert_eq!(found_rank, 1);
            }
            other => panic!("expected RankMismatch, got {:?}", other),
        }
    }

    #[test]
    fn try_get_returns_index_out_of_bounds_error() {
        let t = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
        match t.try_get(&[5, 0]) {
            Err(TensorError::IndexOutOfBounds { index, shape }) => {
                assert_eq!(index, vec![5, 0]);
                assert_eq!(shape, vec![2, 2]);
            }
            other => panic!("expected IndexOutOfBounds, got {:?}", other),
        }
    }

    #[test]
    fn try_get_mut_writes_correct_value() {
        let mut t = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
        *t.try_get_mut(&[0, 1]).unwrap() = 99;
        assert_eq!(*t.try_get(&[0, 1]).unwrap(), 99);
        // Neighboring elements must stay untouched
        assert_eq!(*t.try_get(&[0, 0]).unwrap(), 1);
        assert_eq!(*t.try_get(&[1, 1]).unwrap(), 4);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn index_panics_on_out_of_bounds() {
        let t = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
        let _ = t[&[2, 0][..]];
    }

    #[test]
    #[should_panic(expected = "rank mismatch")]
    fn index_panics_on_rank_mismatch() {
        let t = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
        let _ = t[&[0][..]];
    }

    // ── try_add / try_sub / try_hadamard error handling ─────────────────
    #[test]
    fn try_add_returns_shape_mismatch_error() {
        let a = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
        let b = Tensor::from_vec(&[4], vec![1, 2, 3, 4]);
        match a.try_add(&b) {
            Err(TensorError::ShapeMismatch { expected, found }) => {
                assert_eq!(expected, vec![2, 2]);
                assert_eq!(found, vec![4]);
            }
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn add_operator_panics_on_shape_mismatch() {
        let a = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
        let b = Tensor::from_vec(&[4], vec![1, 2, 3, 4]);
        let _ = &a + &b;
    }

    #[test]
    fn try_sub_returns_shape_mismatch_error() {
        let a = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
        let b = Tensor::from_vec(&[1, 4], vec![1, 2, 3, 4]);
        assert!(matches!(a.try_sub(&b), Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn try_hadamard_returns_shape_mismatch_error() {
        let a = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
        let b = Tensor::from_vec(&[1, 4], vec![1, 2, 3, 4]);
        assert!(matches!(a.try_hadamard(&b), Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    #[should_panic(expected = "Hadamard failed")]
    fn hadamard_panics_on_shape_mismatch() {
        let a = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
        let b = Tensor::from_vec(&[1, 4], vec![1, 2, 3, 4]);
        let _ = a.hadamard(&b);
    }

    #[test]
    fn try_add_matches_existing_add_operator() {
        let a = Tensor::from_vec(&[2, 2], vec![1, 2, 3, 4]);
        let b = Tensor::from_vec(&[2, 2], vec![5, 6, 7, 8]);
        let via_try = a.try_add(&b).unwrap();
        let via_op = &a + &b;
        assert_eq!(via_try.as_slice(), via_op.as_slice());
    }
}
