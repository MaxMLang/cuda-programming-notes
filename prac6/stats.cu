// stats.cu
// CUDA implementation for calculating mean and standard deviation.

#include <cuda_runtime.h>
#include <stdio.h>

// This kernel performs a parallel reduction on a chunk of the data.
// Each block computes a partial sum and sum-of-squares.
__global__ void stats_reduction_kernel(const float* g_data, size_t n, float* g_partial_sums, float* g_partial_sq_sums) {
    
    // Shared memory for this block. Fast memory accessible by all threads in the block.
    extern __shared__ float s_data[];
    float* s_sums = s_data;
    float* s_sq_sums = &s_data[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    // Each thread calculates its initial sum and sum-of-squares.
    // Here we use a grid-stride loop to handle cases where N is not a multiple of blockDim.x
    float my_sum = 0.0f;
    float my_sq_sum = 0.0f;
    while (i < n) {
        float val = g_data[i];
        my_sum += val;
        my_sq_sum += val * val;
        i += gridDim.x * blockDim.x;
    }
    s_sums[tid] = my_sum;
    s_sq_sums[tid] = my_sq_sum;

    __syncthreads(); // Wait for all threads to finish loading into shared memory.

    // Perform the reduction in shared memory.
    // Each step halves the number of active threads.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sums[tid] += s_sums[tid + s];
            s_sq_sums[tid] += s_sq_sums[tid + s];
        }
        __syncthreads(); // Synchronize after each reduction step.
    }

    // Thread 0 of each block writes the block's final result back to global memory.
    if (tid == 0) {
        g_partial_sums[blockIdx.x] = s_sums[0];
        g_partial_sq_sums[blockIdx.x] = s_sq_sums[0];
    }
}


// This is the bridge function called by the host.
extern "C"
void calculate_stats_gpu(const float* h_data, size_t n, float* mean, float* std_dev) {
    
    // --- 1. Allocate Memory on GPU ---
    float *d_data, *d_partial_sums, *d_partial_sq_sums;
    cudaMalloc(&d_data, n * sizeof(float));

    int threads = 256;
    int blocks = (n + threads - 1) / threads; // Calculate number of blocks needed
    
    cudaMalloc(&d_partial_sums, blocks * sizeof(float));
    cudaMalloc(&d_partial_sq_sums, blocks * sizeof(float));

    // --- 2. Copy Data from Host to Device ---
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    // --- 3. Launch Kernel ---
    // The third argument is the dynamic shared memory size per block.
    // We need space for two float arrays of size `threads`.
    unsigned int shared_mem_size = threads * 2 * sizeof(float);
    stats_reduction_kernel<<<blocks, threads, shared_mem_size>>>(d_data, n, d_partial_sums, d_partial_sq_sums);

    // --- 4. Copy Partial Results Back to Host ---
    float* h_partial_sums = (float*)malloc(blocks * sizeof(float));
    float* h_partial_sq_sums = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial_sums, d_partial_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_partial_sq_sums, d_partial_sq_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // --- 5. Final Reduction on CPU & Calculation ---
    // (This final sum is small, so doing it on the CPU is fine)
    double total_sum = 0.0;
    double total_sq_sum = 0.0;
    for (int i = 0; i < blocks; i++) {
        total_sum += h_partial_sums[i];
        total_sq_sum += h_partial_sq_sums[i];
    }
    
    *mean = static_cast<float>(total_sum / n);
    *std_dev = static_cast<float>(sqrt((total_sq_sum / n) - (*mean) * (*mean)));

    // --- 6. Free Memory ---
    free(h_partial_sums);
    free(h_partial_sq_sums);
    cudaFree(d_data);
    cudaFree(d_partial_sums);
    cudaFree(d_partial_sq_sums);
}