/// Minimum number of elements before switching to parallel execution.
/// Below this threshold the thread-spawning overhead outweighs the gain.
/// 32×32 = 1024 is a practical lower bound for f64 element-wise ops.
pub const PAR_THRESHOLD: usize = 1024;

/// Returns `true` if the workload is large enough to benefit from parallelism.
#[inline]
pub fn should_parallelize(num_elements: usize) -> bool {
    num_elements >= PAR_THRESHOLD
}
