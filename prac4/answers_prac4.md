 ### Practical 4 - Reduction Notes

#### Tasks 1 & 2 - Initial Code Analysis
* The initial `reduction.cu` file was compiled and run. It worked correctly for the default 512 threads.
* The core logic uses dynamically allocated shared memory. 
* The algorithm is a classic binary tree reduction: all threads load one element into shared memory, then half the threads add values from the other half in a loop.
* `__syncthreads()` is essential here to prevent race conditions between each step of the reduction. 

---

#### Task 3 - Handling Non-Power-of-Two Block Sizes
* The initial code fails for block sizes that aren't a power of two (like 192) because the `d=d/2` logic assumes it. 
* The solution is to find the largest power of 2 less than the block size (e.g., 128 for 192).
* The "extra" threads (128-191) must first add their values to the beginning of the shared memory array (e.g., thread 128 adds to thread 0's value). 
* After a `__syncthreads()`, the normal binary reduction can proceed on the power-of-two portion.

---

#### Task 4 - Multi-Block Reduction
* To handle arrays larger than a single block, the code was modified to use multiple blocks. 
* The strategy used was the first one suggested in the lecture notes: 
    * Each block calculates a partial sum for its chunk of the input data.
    * Each block writes its partial sum to a unique spot in a temporary output array in global memory. 
    * The main `main()` function copies this array of partial sums back to the CPU and performs the final summation there.
* The alternative, using atomic operations on a single memory location, was not implemented but noted. 

---

#### Task 5 - Shuffle Instruction Optimization
* The block-level reduction was optimized using warp-level shuffle instructions as suggested. 
* The final kernel (`reduction_shuffle_kernel`) uses a two-stage approach:
    1.  Warp Reduction --> A very fast reduction is performed inside each 32-thread warp using `__shfl_down_sync`. This avoids using shared memory or `__syncthreads()` for this stage.
    2.  Block reduction --> The leader thread of each warp writes its partial sum to shared memory. Then, the first warp performs a final reduction on these few values (one per warp) to get the block's total sum.

---

#### Key Finding: The Floating-Point Precision Bug
* The first multi-block test runs showed a massive error (`~1.2e6`) when using `float`.
* This was not a logic error. It was a numerical precision issue. The `float` data type doesn't have enough precision to accurately sum millions of numbers without accumulating significant errors. The different order of operations between the CPU and GPU made the error obvious.
* Solution --> The code was modified to use `double` for all data and calculations. This provides much higher precision.
* After switching to `double`, the reduction error became `0.0`, confirming the problem was numerical, not logical.

---

#### Task 6 - Application to Laplace Solver
* The new global reduction kernel was used to compute the root-mean-square (RMS) change in the `laplace3d` code. 
* This required a three-step process inside the main iteration loop:
    1.  Run the `GPU_laplace3d` kernel as normal to get the updated grid `d_u2`.
    2.  Run a new kernel, `compute_squared_change`, that calculates `(d_u2[i] - d_u1[i])^2` for every point and saves it to a temporary array.
    3.  Run our new `reduction_shuffle_kernel` on the temporary array to get the total sum of squared changes. 
* The final RMS value was calculated on the CPU using this sum. The result after 50 iterations was a plausible `0.002299`.

---

#### BONUS - Particle Filter Benchmark

* Added `particle_filter_benchmark.cu` - implements a 1D particle filter with two different scan algorithms for the resampling step
* Uses ~1 million particles (2^20) to benchmark at scale

---

#### Particle Filter Basics

* Each particle has position $x$ and weight $w$
* Three main steps per iteration:
  1. Predict: $x_{t+1} = x_t + \mathcal{N}(0, 0.1)$ (add Gaussian noise)
  2. Weight: $w_i = \exp\left(-\frac{(x_i - z)^2}{2\sigma^2}\right)$ where $z$ is measurement, $\sigma=0.5$
  3. Resample: select new particles based on normalized CDF of weights

---

#### Two Scan Implementations

* Scan is needed to compute CDF for resampling
* Version 1: Blelloch scan (shared memory)
  - Classic work-efficient parallel prefix sum
  - Up-sweep phase builds partial sums
  - Down-sweep distributes prefix values
  - Uses 2x shared memory per block
* Version 2: Warp shuffle optimization
  - Intra-warp scan using `__shfl_up_sync`
  - No shared memory needed within warps
  - First warp scans the per-warp sums
  - More register pressure but less shared memory

---

#### Key Fix: Binary Search Resampling

* Original linear search caused kernel timeouts with 1M particles
* Replaced with parallel binary search (lower_bound)
* Each thread independently finds its particle in O(log n) time
* Prevents the massive slowdown from sequential scanning

---

#### Performance Results

* Blelloch scan: 11.885 ms for 100 iterations (0.119 ms/iter)
* Warp shuffle: 10.965 ms for 100 iterations (0.110 ms/iter)
* ~8% speedup from warp shuffle optimization
* Both versions handle 1M particles without timeout issues