/// Test-only helper for comparing floating point values with tolerance.
/// `assert_eq!` on f64 is fragile due to rounding error — this is the
/// safe alternative used throughout NumRS's test suite.
#[cfg(test)]
#[macro_export]
macro_rules! assert_close {
    ($a:expr, $b:expr) => {
        assert_close!($a, $b, 1e-9)
    };
    ($a:expr, $b:expr, $eps:expr) => {
        let (a, b) = ($a, $b);
        assert!(
            (a - b).abs() < $eps,
            "assert_close failed: {} vs {} (diff = {}, eps = {})",
            a, b, (a - b).abs(), $eps
        );
    };
}
