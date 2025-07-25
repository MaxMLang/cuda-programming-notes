### CUDA Concepts Overview

-   Thread
    -   Smallest unit of work.
    -   Like one person doing one tiny task.
    -   Has its own private memory (registers).

-   Block
    -   A group of threads, like a team.
    -   Threads in a block can share data using fast 'shared memory'.
    -   Can also sync up so they all wait for each other.

-   Warp
    -   A hardware thing, not a programming thing.
    -   The GPU executes threads in groups of 32, called a warp.
    -   All 32 threads in a warp do the exact same instruction at the same time.

-   Hierarchy
    -   You write code for a thread.
    -   You group threads into blocks.
    -   The hardware groups threads into warps for execution.

---

### Additional Notes

#### Bandwidth vs. Compute Limited

-   What is it?
    -   Bandwidth-limited means your program is slow because it's waiting for data to be moved around (memory is the bottleneck). Also called memory-bound.
    -   Compute-limited means your program is slow because the math is too hard and the GPU cores can't keep up (processing power is the bottleneck). Also called compute-bound.

-   Why does it matter?
    -   You need to know what's slowing you down to fix it.
    -   If bandwidth-limited, you need to:
        -   Move less data.
        -   Use shared memory more.
        -   Make memory access patterns better.
        -   Basically, stop waiting for data.
    -   If compute-limited, you need to:
        -   Make your math calculations more efficient.
        -   Do more math at the same time (in parallel).
        -   Avoid things that slow down calculations, like warp divergence.
        -   Basically, make the math faster.
    -   Optimizing the wrong thing is a waste of time. If you're stuck waiting on memory, making the math faster won't help.


### Warp Shuffles

-   What are they?
    -   A way for threads in the same warp to trade data directly.
    -   It's faster than writing to shared memory and having another thread read it back.

-   Why are they useful?
    -   Speed! Avoids the round-trip to shared memory for intra-warp communication.
    -   Super helpful for parallel patterns like reductions and scans where threads need to aggregate their neighbors' data.
    -   The `_shfl_xor_sync` one is great for "butterfly" operations, which are common in reductions and FFTs.

-   Types of Shuffles
    -   `_shfl_up_sync`: get data from a thread with a lower ID.
    -   `_shfl_down_sync`: get data from a thread with a higher ID.
    -   `_shfl_xor_sync`: get data from a thread whose ID is an XOR of your ID.
    -   `_shfl_sync`: get data from a specific thread by its lane ID.

-   Important Gotcha
    -   You can only get data from a thread that is also participating in the shuffle.
    -   If you try to shuffle with an inactive thread (e.g., one that's sitting out because of an `if` statement), you'll get junk data. This means you have to be careful with conditional code.

---

### Reduction Operations

-   What is it?
    -   Taking a big array of numbers and "reducing" it to a single value.
    -   Think summing everything, or finding the single max/min value in an array.
    -   The operation needs to be associative and commutative (like addition) so you can do it in any order.

-   How it works in CUDA (2-step process)
    -   1.  Local Reduction: Each thread block calculates its own partial sum (or min/max) from a chunk of the data.
        -   This is done in parallel inside the block. Threads pair up to add values, then those results are paired up, and so on, until there's one value left.
        -   Uses fast shared memory to hold the intermediate values.
        -   You need `__syncthreads()` between each step to make sure threads don't read old data.
        -   You can speed up the final steps within a block by using warp shuffles once only 32 threads (a single warp) are left working.

    -   2.  Global Reduction: Add up the partial sums from all the different blocks.
        -   Option A: Copy all the partial sums from the GPU back to the CPU and just add them there. Simple, but involves data transfer.
        -   Option B: Use an `atomicAdd` operation. The first thread in each block atomically adds its partial sum to a single variable in global memory.

---

### Scan Operations (Prefix Sum)

-   What is it?
    -   It's an operation that looks sequential, but isn't.
    -   Given an array `[a, b, c, d, ...]`, the output is `[0, a, a+b, a+b+c, ...]`. Each output element is the sum of all the input elements that came before it.
    -   Super important for things like sorting algorithms.

-   How it works in CUDA
    -   It's complex! The main advice is to find a pre-built library function (like in Thrust) unless you're an expert.
    -   The general idea is similar to reduction: do work locally in blocks, then combine the block results.

-   The process:
    -   1.  Local Scan: Each block computes a scan on its own chunk of the data.
        -   An efficient way to do this involves warp shuffles. You first do a scan inside each warp, then do a scan of the warp sums, and then add the offsets back.

    -   2.  Global Scan: This is the tricky part. Each block needs to add the total sum of all the preceding blocks to its local scan results.
        -   But blocks don't run in a guaranteed order (`block 5` might finish before `block 2`), which can cause a deadlock.
        -   The solution is to force an order. You can use an atomic counter to assign each block a unique, sequential ID as it starts. Blocks then have to wait their turn to add their total to a global sum before the next block can proceed. This avoids the deadlock.

#### Tensor Cores - What/Where are they?
- They're basically special hardware units inside each SM.
- Built to do one thing super fast: matrix-multiply-accumulate (MMA), which is $D = A \times B + C$. It's like a FMA but for whole matrices.

- How they work: they take small, low-precision matrices (like fp16), multiply them, and add the result to a bigger, high-precision matrix (like fp32).

- Performance: this is where the big speedup comes from. You do a ton of math per clock cycle, but have to accept lower precision. The A100 graph from the slides showed a huge difference between Tensor Core speeds and standard fp32.

- TF32: a newer format on Ampere+ GPUs. It's a compromise. Has the same range as fp32 but less precision (10 bits instead of 23). Good default for AI stuff.

---

#### Important Libraries
- The lecture covered a bunch of these. The main takeaway is to not reinvent the wheel. If there's a library for what you're doing, just use it. It's already optimized and maintained by NVIDIA.

- Some useful ones:
  - cuBLAS - for dense matrix math.
  - cuFFT - for fourier transforms.
  - cuRAND - for random numbers.
  - cuSPARSE - for sparse matrix math.
  - Thrust - a C++ template library, feels like writing normal C++ (STL-style) but runs on the GPU.
  - NCCL - for multi-GPU / multi-node communication.

---

#### The "Seven Dwarfs"
- A list of the 13 most common patterns in scientific computing.
- Helps you figure out what kind of problem you have and which library to use.
- e.g. if you're doing dense linear algebra, you use cuBLAS. If you're doing spectral methods, you use cuFFT. Monte Carlo methods use cuRAND, etc.

---

#### Debugging Tools
- `compute-sanitizer` is the main tool to remember. It's a lifesaver for finding tricky bugs.
- `--tool memcheck` is the most important one. Catches out-of-bounds errors.
- `--tool racecheck` finds shared memory race conditions.
- `--tool initcheck` finds if you're reading memory you never initialized.
- `--tool synccheck` for when you mess up `__syncthreads()`.

---
### Blocking vs. Asynchronous Host Code

-   Why would you block? (Synchronous)
    -   It's simple and guarantees correctness. You know a task (e.g., a memory copy) is 100% finished before the next line of CPU code runs.
    -   Good for debugging or when the CPU absolutely needs the result from the GPU right away.

-   Why run asynchronously? (Non-blocking)
    -   Performance! The whole point is to overlap operations to hide latency.
    -   While the GPU is busy with a kernel, the CPU can do other work, like preparing the next chunk of data.
    -   Like a busy chef who starts prepping a salad while the main course is in the oven. You get more done in the same amount of time.
    -   This is how you keep both the CPU and GPU busy instead of having one wait for the other.

---

### CUDA Streams

-   What are they?
    -   A stream is just a sequence of GPU commands that are guaranteed to run in the order you issue them (First-In, First-Out). To-do list for a GPU.
    -   By default, everything runs in one "default stream".

-   How they work for performance
    -   You can create multiple streams. Operations in *different* streams can run at the same time (concurrently).
    -   This is the mechanism for overlapping. For our stats example:
        -   Break the data into 4 chunks.
        -   Stream 1: `Copy chunk 1 -> Kernel on chunk 1 -> Copy result 1 back`
        -   Stream 2: `Copy chunk 2 -> Kernel on chunk 2 -> Copy result 2 back`
        -   And so on for streams 3 and 4.
        -   The GPU can be running the kernel for chunk 1 while it's also using its copy engine to transfer chunk 2.
    -   To make this work, you need to use `cudaMemcpyAsync()` and allocate your host memory as **pinned memory** (using `cudaHostAlloc()`). This tells the OS not to move the memory around, so the GPU knows its exact physical address for direct transfers.

---

### Multi-GPU Computing

-   General idea
    -   Use `cudaSetDevice(id)` to select which GPU you want a command to run on.
    -   Each GPU is its own little world with its own memory. You have to explicitly manage where your data lives.
    -   When you run the same kernel on two GPUs instead of one, the execution time is roughly halved, as shown in the lecture example.

-   GPUDirect
    -   A fancy name for tech that lets GPUs talk directly to other hardware (other GPUs, network cards, storage) without needing the CPU to be a middleman.
    -   This cuts out slow memory copies through CPU RAM, lowering latency and freeing up the CPU.
    -   **GPUDirect P2P** is the most common one. It lets a kernel on GPU 0 directly read, write, or even perform atomic operations on memory that lives on GPU 1. Needs a fast connection between the GPUs like NVLink to be effective.

-   Multi-GPU with OpenMP
    -   This is the simplest approach for a single server with multiple GPUs. It's a shared-memory model.
    -   You use an OpenMP `#pragma omp parallel` block to create several CPU threads.
    -   Inside the block, each CPU thread grabs its own GPU, e.g., thread `i` calls `cudaSetDevice(i)`.
    -   Each thread then gets its own independent default stream to work with, so they don't interfere with each other.

-   Multi-GPU with NVSHMEM & MPI
    -   This is for big distributed systems, like supercomputers with many nodes.
    -   **MPI** is the traditional way. Each MPI process (usually a CPU core) controls one GPU. To share data, the CPU has to post an `MPI_Send` and `MPI_Recv`, which orchestrates the data movement. It works, but the CPU is heavily involved.
    -   **NVSHMEM** is the newer, faster way. It's based on OpenSHMEM. Instead of the CPU, the GPUs can directly "put" data into another GPU's memory across the network. This is faster because the GPU initiates the transfer itself, cutting out the CPU overhead.

-   Quick Comparison Table

| Approach | Scale | Programming Model | Data Movement |
| :--- | :--- | :--- | :--- |
| **OpenMP + CUDA** | Single Machine | Shared Memory | CPU threads control GPUs |
| **MPI + CUDA** | Multi-Node Cluster | Message Passing | CPU orchestrates data transfer |
| **NVSHMEM** | Multi-Node Cluster | PGAS (Global Address Space) | GPU initiates data transfer |