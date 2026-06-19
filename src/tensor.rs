use std::fmt;
use std::ops::{Add, Sub, Mul, Index, IndexMut};
use colored::Colorize;
use num_traits::{Zero, One};

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
