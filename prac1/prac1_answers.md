### Practical 1: Getting Started Notes

This practical is a basic introduction to CUDA programming. The main goals are to understand the relationship between the host (CPU) and the device (GPU), how to copy data between them, and how to implement proper error-checking.

***

### `prac1a` vs. `prac1b`: Error Checking is Key

The primary difference between `prac1a.cu` and `prac1b.cu` is that the 'b' version includes robust error checking.

* `prac1a.cu` is the minimal code. If a problem occurs, like a memory allocation failure, it might crash without a clear explanation.
* `prac1b.cu` incorporates helper functions like `checkCudaErrors()` and `getLastCudaError()`. These functions act as wrappers that immediately identify and report if a CUDA function or a kernel launch fails, which is crucial for debugging.
* For instance, when intentionally breaking the code by setting `nthreads = 10000` (an invalid number), `prac1b` provides a specific error message about the invalid launch configuration, whereas `prac1a` fails without such clear guidance.

***

### Thread Indexing & Kernel Output

The kernel in the initial files calculates a global thread ID (`tid`) but assigns a value to an array based on the local thread ID (`threadIdx.x`).

`int tid = threadIdx.x + blockDim.x*blockIdx.x;`
`x[tid] = (float) threadIdx.x;`

* With 2 blocks and 8 threads each, the `tid` (global ID) ranges from 0 to 15.
* However, `threadIdx.x` (the local ID within a block) ranges from 0 to 7 and then **resets** to 0-7 for the second block.
* This behavior explains why the output shows the values `0.0, 1.0, ..., 7.0` repeating for the second block of threads (indices 8-15).

Adding a `printf` statement inside the kernel is effective. It's important to note that, as mentioned in the practical's instructions, the output from the GPU is buffered. This can cause messages to appear on the screen in an order that doesn't perfectly match the execution sequence.

***

### Task 7: Vector Addition

I modified the `prac1b.cu` file to add two vectors (`a` and `b`), storing the result in a third vector (`c`). The process involved these key steps:
1.  **Host Memory**: Allocate three arrays on the host CPU (`h_a`, `h_b`, `h_c`) with `malloc`.
2.  **Device Memory**: Allocate three corresponding arrays on the GPU device (`d_a`, `d_b`, `d_c`) with `cudaMalloc`.
3.  **Copy to Device**: Transfer the input arrays (`h_a` and `h_b`) to their device counterparts (`d_a` and `d_b`) using `cudaMemcpy`.
4.  **Execute Kernel**: Launch the `vector_add_kernel` on the GPU. This kernel performs the simple operation: `c[tid] = a[tid] + b[tid];`.
5.  **Copy from Device**: Retrieve the result array `d_c` from the GPU and copy it back to the host's `h_c` array.
6.  **Cleanup**: Deallocate all the memory on both the device (`cudaFree`) and the host (`free`) to prevent memory leaks.

***

### `prac1c`: A Look at Unified Memory

The final part of the practical explored `prac1c.cu`, which utilizes "managed memory."

* This approach significantly simplifies the code. You only need to allocate a single pointer with `cudaMallocManaged`, rather than separate pointers for the host and device.
* You don't need any explicit `cudaMemcpy` calls. The CUDA driver intelligently handles the migration of data between the CPU and GPU as it's needed.
* It is still crucial to call `cudaDeviceSynchronize()` after launching the kernel. This function pauses the host's execution, forcing it to wait for the GPU to complete its calculations before proceeding to read the results, thereby ensuring data consistency.