### Practical 8 - Scan and Recurrence Notes

#### Initial Code and Multi-Block Scan (Task 2)

* First step was extending the single-block scan to handle large arrays (1,000,000 elements).
* This needed a three-kernel approach:
    1.  A `block_scan` kernel runs on the whole array. Each block does a local scan and writes its total sum to a separate `d_block_sums` array.
    2.  The `block_scan` kernel runs again, but this time on the small `d_block_sums` array to get the prefix sum of the block sums.
    3.  A final `add_block_sums` kernel adds the result from step 2 back to the intermediate results from step 1.
* This makes a correct, scalable scan for any input size.

#### Key Finding: The Floating-Point Precision Bug

* The `scan_multiblock` output had a massive RMS error (~2.1e6).
* This isn't a logic bug. It's a numerical precision problem with `float`.
* When summing millions of numbers, the order of operations really matters because of rounding errors. The parallel scan adds numbers in a different order than the sequential CPU `scan_gold` function.
* This difference accumulates into a huge final error.
* The fix, like in the reduction practical, is to switch from `float` to `double` for better precision on large datasets.

#### Shuffle-Based Scan Optimization (Task 3)

* The `scan_shuffle` kernel was made to optimize the scan *inside* a single block.
* It uses `__shfl_up_sync` for a fast scan within each 32-thread warp, avoiding shared memory for that part.
* Shared memory is still needed for communication *between* warps. The last thread of each warp writes its sum to shared memory, and the first warp scans those values.
* The final error for this version (~755) was smaller but not zero. This is because the test only had 512 elements, so the float error didn't build up as much. The error is still there because the shuffle adds numbers in yet another different order.

#### Recurrence Equation Scan (Task 4)

* The last task was changing the scan to solve a first-order recurrence: $y_n = a_n \times y_{n-1} + b_n$.
* The main idea is that this is still an associative operation, so the parallel scan works if you just redefine the "plus" operator.
* A `float2` was used to hold the coefficient pairs $(a, b)$.
* A `__device__` function, `recurrence_op`, was made to define how to combine two pairs. If you combine $(a_1, b_1)$ and $(a_2, b_2)$, the new pair is $(a_2a_1, a_2b_1+b_2)$.
* The scan kernel is almost the same as the basic one, but it calls `recurrence_op` instead of `+`.
* The output showed very small errors, which confirms the logic is correct.