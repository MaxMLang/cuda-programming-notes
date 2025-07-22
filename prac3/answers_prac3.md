### Practical 3 - Laplace Solver Notes

#### Q1 & Q2 - First Run & Code Check

* Compiled everything with the `Makefile`. It created a `bin/` directory, which is tidy.
* Ran the original `laplace3d` on the 512x512x512 grid.
* CPU (Gold) time was slow, ~30 seconds.
* GPU time was 99.5 ms for 20 iterations, so about **4.98 ms per iteration**. The GPU is obviously much faster.
* RMS error was 0.000000, so the GPU result is correct.
* The code uses pointer swapping for `u1` and `u2` in the main loop. This is a key optimization to avoid slow `cudaMemcpy` calls between iterations.

#### Q3 & Q4 - Modifying for Benchmarking

* To properly test performance, the code needs to be run on a bigger 1024x1024x1024 grid.
* The `laplace3d_benchmark` program handles this automatically (it's hardcoded to 1024^3 and doesn't run the slow CPU part).

#### Q5 - Optimizing the Kernels

The main task is to find the best block size for the two different kernel versions.

* **v1 (original kernel):** Uses a 2D grid of threads. Each thread loops over the z-dimension.
* **v2 (new kernel):** Uses a 3D grid. Each thread handles only one point.

An important note from the compiler output:
* The v1 kernel used **56 registers**.
* The v2 kernel only used **21 registers**.
* Using fewer registers is a big deal. It means the GPU can run more threads at once (higher occupancy), which helps hide memory latency and should make the v2 kernel faster.

##### Benchmark Results (1024^3 grid)

I ran `make benchmark` to get these numbers.

**v1 Kernel (Original) Performance**
| Block Dim (X, Y) | Total Threads | Time per iter (ms) |
| :--- | :--- | :--- |
| 16 x 16 | 256 | 37.423 |
| **32 x 8** | **256** | **33.413** |
| 8 x 32 | 256 | 66.679 |
| 32 x 16 | 512 | 34.190 |

**v2 Kernel (New) Performance**
| Block Dim (X, Y, Z) | Total Threads | Time per iter (ms) |
| :--- | :--- | :--- |
| **8 x 8 x 8** | **512** | **17.765** |
| 4 x 8 x 16 | 512 | 24.791 |
| 4 x 16 x 16 | 1024 | 26.183 |
| 8 x 8 x 16 | 1024 | 19.976 |

##### Conclusion
The best time for the **v1 kernel** was **33.413 ms** with a 32x8 block. The best time for the **v2 kernel** was **17.765 ms** with an 8x8x8 block.

The v2 kernel is almost **twice as fast** as the v1 kernel. The lower register count and higher parallelism of the v2 approach are clearly more efficient on the Volta GPU.

---
#### Q6 - Profiling with `ncu`

You need to run the NVIDIA profiler on the original executables to count instructions.

*Commands:*
```bash
ncu --metrics "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum" ./bin/laplace3d

ncu --metrics "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum" ./bin/laplace3d_new
````

*Predicted Results:*

  * **Floating Point Ops:** The `fp32` count will be almost identical for both versions. They do the same math.
  * **Integer Ops:** The `integer` count for `laplace3d_new` (v2) will be much higher. This is because every thread calculates its full 3D index (`i,j,k`), while v1 threads only calculate `i,j` once and then do a cheap increment in a loop. This confirms why v2 needs fewer registers - the calculation is simpler and doesn't need a loop counter.

-----

#### Q7 - Memory Bandwidth

Let's calculate the bandwidth for the fastest run, which was the **v2 kernel** with an 8x8x8 block on the 1024^3 grid.

  * Grid: 1024 x 1024 x 1024 (N = 1,073,741,824 points)

  * Data per point: 6 reads + 1 write = 7 floats.

  * Total data per iter: `N * 7 * 4 bytes` = **30.06 GB**

  * Time per iter: **17.765 ms** (or 0.017765 s)

  * **Effective Bandwidth = 30.06 GB / 0.017765 s ≈ 1692 GB/s**

This number is incredibly high, much higher than the V100's peak DRAM bandwidth of \~900 GB/s. This proves two things:

1.  The application is **bandwidth-limited**. The bottleneck is moving data, not doing the math.
2.  The GPU **cache** is working extremely well. Most of the 6 reads for the stencil are being served by the fast L1/L2 cache, not the slow main memory. This is why the calculated bandwidth is so high; it reflects the logical data movement, which is much greater than the physical data movement from DRAM.

