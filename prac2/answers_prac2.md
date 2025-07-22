### Practical 2 - Monte Carlo Notes

#### Q1 & Q2 - First look at the code

* The code uses `__constant__` memory for simulation parameters like `r`, `sigma`, and `T`. [cite: 8] This is a small, fast, read-only cache. When all threads in a warp read the same constant, it's broadcast to all of them at once, which is super efficient. [cite: 4]
* It uses `cudaEvent_t` for timing. [cite: 5] The `cudaEventSynchronize(stop)` call is important; it makes the CPU wait for the GPU to actually finish before stopping the clock. [cite: 11, 12]
* My first run of the default `prac2.cu` code took **11.59 ms** for the main kernel.

#### Q3 & Q4 - Memory Access Patterns - Coalesced vs. Strided

This is the main point of the practical: the difference between how Version 1 and Version 2 read from the big `d_z` array of random numbers.

* **Version 1 (Coalesced)** is fast because threads in a warp access memory right next to each other. The hardware can grab all the data in one big chunk. This is **coalesced memory access**. [cite: 17]
* **Version 2 (Strided)** is slow because threads access memory with big gaps in between. The hardware has to do lots of separate, slow memory fetches. This is **strided memory access** and it kills performance. [cite: 17]

##### Benchmark Results

I ran the `prac2_benchmark` script to compare them. The results clearly show the impact of the memory access pattern.

| Memory Access | Kernel Time (ms) | Performance Factor |
| :--- | :--- | :--- |
| **Version 1 (Coalesced)** | 10.18 ms | 1x |
| **Version 2 (Strided)** | 246.55 ms | **24.2x slower** |

The strided version is over 24 times slower, proving that coalesced access is critical for performance.

---
#### Q5 - Memory Bandwidth


* **Paths:** `NPATH` = 9,600,000
* **Timesteps:** `N` = 100
* **Data read per path:** `2 * 100` floats
* **Total Data Read:** `9,600,000 * 200 * 4 bytes` = **7.68 GB** [cite: 21]
* **Kernel Time:** **10.18 ms** (0.01018 s)

Bandwidth = 7.68 GB / 0.01018 s ≈ **754.4 GB/s**

Very high number and a large fraction of the theoretical peak bandwidth of the Volta V100 GPU (~900 GB/s). It shows the kernel is very efficient at reading memory because the access is coalesced. [cite: 22]

---
#### Q6 - Custom `az^2+bz+c` program

Here's the code to solve this. It uses constant memory for `a`, `b`, and `c` and averages the result on the host, as suggested in the practical.  The theoretical average should be `a+c`. 

```cuda
// To compile: nvcc prac2_q6.cu -o ./bin/prac2_q6 -lcurand -arch=sm_70
#include <stdio.h>
#include <curand.h>
#include <helper_cuda.h>

// Use constant memory for a, b, c
__constant__ float A, B, C;

__global__ void avg_kernel(float* d_rand, float* d_results) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int N_PER_THREAD = 200;

    // Point to this thread's block of random numbers
    float* my_rands = d_rand + id * N_PER_THREAD;

    float sum = 0.0f;
    for (int i = 0; i < N_PER_THREAD; i++) {
        float z = my_rands[i];
        sum += A * z * z + B * z + C;
    }

    // Write the average for this thread to the results array
    d_results[id] = sum / N_PER_THREAD;
}

int main() {
    int n_threads = 1024;
    int n_per_thread = 200;
    int total_rands = n_threads * n_per_thread;

    // Set constants
    float h_A = 2.0f, h_B = 3.0f, h_C = 4.0f;
    cudaMemcpyToSymbol(A, &h_A, sizeof(float));
    cudaMemcpyToSymbol(B, &h_B, sizeof(float));
    cudaMemcpyToSymbol(C, &h_C, sizeof(float));

    // Allocate memory
    float* d_rand;
    float* d_results;
    float* h_results = (float*)malloc(n_threads * sizeof(float));
    cudaMalloc(&d_rand, total_rands * sizeof(float));
    cudaMalloc(&d_results, n_threads * sizeof(float));

    // Generate random numbers
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, d_rand, total_rands, 0.0f, 1.0f);

    // Run kernel
    avg_kernel<<<n_threads / 128, 128>>>(d_rand, d_results);

    // Copy results back and average on host
    cudaMemcpy(h_results, d_results, n_threads * sizeof(float), cudaMemcpyDeviceToHost);

    double final_avg = 0.0;
    for (int i = 0; i < n_threads; i++) {
        final_avg += h_results[i];
    }
    final_avg /= n_threads;

    printf("Monte Carlo average: %f\n", final_avg);
    printf("Theoretical average (a+c): %f\n", h_A + h_C);

    // Cleanup
    curandDestroyGenerator(gen);
    free(h_results);
    cudaFree(d_rand);
    cudaFree(d_results);
    cudaDeviceReset();
    return 0;
}