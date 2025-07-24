### Practical 7 - Tridiagonal Solver Notes

#### Main Goal
* Learn how to solve a tridiagonal system of equations on the GPU.
* This kind of problem comes up in physics simulations, like heat diffusion
* The main equation is $A~x^{n+1}=\lambda~x^{n}$. The CPU version is sequential, so the goal is to parallelize it.

---

#### The CPU `gold_trid` code
* Uses the standard Thomas Algorithm.
* It's a two-stage process: a "forward pass" followed by a "reverse pass".
* This is inherently sequential because each calculation depends on the previous one. Slow.

---

#### The GPU `trid` kernel & PCR
* The GPU version uses a parallel algorithm called **Parallel Cyclic Reduction (PCR)**
* It's totally different from the CPU version. Instead of a forward/backward pass, all threads work at once.
* In a loop (`nt=1, 2, 4, 8...`), each thread communicates with its neighbours at an increasing distance `nt`.
* This eliminates variables in parallel until every thread has its final answer.
* `__syncthreads()` is super important. It's used as a barrier to make sure all threads finish a step before anyone starts the next one. This stops threads from reading old data.

---

#### Task 3 - Dynamic Shared Memory
* The first version of the code had `__shared__ float a[128]`, which meant it could only handle problems up to size 128
* Switched to dynamic shared memory
    * Changed the declaration to `extern __shared__ float s_mem[]`.
    * Passed the required size during the kernel launch: `GPU_trid<<<..., ..., shmem_size>>>`.
    * Inside the kernel, had to set up pointers manually to divide `s_mem[]` into the `a`, `c`, and `d` arrays.
* This makes the code flexible, now it can run with `NX=256` or anything else.

---

#### Task 4 - Multiple Independent Systems
* Modify the code to solve many separate tridiagonal systems at the same time.
* Launch `M` thread blocks to solve `M` problems
* Each block runs the same PCR algorithm, but on a different chunk of data.
* The key change in the kernel was using `blockIdx.x` to figure out which problem a thread belongs to.
* This means global memory access had to be indexed like `u[blockIdx.x * NX + threadIdx.x]`.
* This is a really common and effective pattern for getting high throughput on a GPU.