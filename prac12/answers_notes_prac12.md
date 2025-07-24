### Practical 12 - Streams Notes

#### Task 1 - Overlapping Kernels (`kernel_overlap.cu`)

* The first part of the code launches a bunch of small kernels into the default stream.
* Because the default stream is synchronous (FIFO), the kernels run one after the other, even though no single kernel uses the whole GPU.
* The second part creates multiple streams and launches one kernel into each.
* This lets the GPU scheduler run the kernels concurrently, which gave a massive speedup (from ~46ms to ~0.016ms). This is great for when you have lots of independent, small tasks.

---

#### Task 2 - Overlapping Transfers and Kernels (`work_streaming.cu`)

* The main goal here was to fix a common inefficient pattern:
    1.  Copy ALL data to GPU
    2.  Process ALL data
    3.  Copy ALL results back
* This leaves the GPU compute units idle during copies, and the copy engines idle during computation.
* The solution (`work_streaming_solution.cu`) was to create a pipeline:
    * Broke the big 1GB dataset into smaller chunks (8 in this case).
    * Created 8 CUDA streams.
    * **Crucially**, used `cudaMallocHost` to create **pinned memory** on the host. This is required for async copies to actually be asynchronous.
    * Used `cudaMemcpyAsync` instead of the normal `cudaMemcpy`.
* The main loop then queues up operations for each chunk in its own stream:
    1.  `cudaMemcpyAsync` (Host to Device)
    2.  `do_work` kernel launch
    3.  `cudaMemcpyAsync` (Device to Host)
* Because these are in different streams, the GPU can start copying chunk `i+1` while it's computing chunk `i`. This overlap cut the total runtime in half (from ~240ms to ~120ms).

---

#### Task 3 - Statistics Application (`bootstrap_ci.cu`)

* Created an example to calculate bootstrap confidence intervals. This involves generating thousands of resamples from a dataset and calculating their means.
* This is a good use case for streams because each resample is independent.
* The code divides the total number of resamples across the streams. Each stream's kernel calculates a subset of the bootstrap means.
* This allows the computation for all streams to happen in parallel, speeding up the process.

---

#### Key Bugs & Fixes

* **Compiler Error in `bootstrap_ci.cu`**: Had a `float` variable named `time` which conflicted with the C++ standard library function `time()`. Renaming the variable to `elapsedTime` fixed it.
* **`no kernel image is available` error**: This happened when running on the cluster. It means the code was compiled for a specific GPU architecture (e.g., Turing) but the job ran on a node with a different one (e.g., Volta).
* **Fix**: Updated the `Makefile` to build a "fat binary" by including `-gencode` flags for multiple architectures (`sm_60`, `sm_70`, `sm_75`, `sm_80`). This makes the executable larger but much more portable.
* **Incorrect `bootstrap_ci` result**: The confidence interval was `[0.0000, 0.0000]`. This was caused by passing the wrong arguments to the kernel. Fixed by making sure the kernel launch matched the kernel's function definition.