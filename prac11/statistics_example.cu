/*
Statistics Example: Parallel Bootstrapping with CUDA Streams.

Bootstrapping is a resampling method used to estimate statistics on a population
by sampling a dataset with replacement. It's an "embarrassingly parallel"
problem because each bootstrap sample is independent.

This code calculates the mean of thousands of bootstrap samples in parallel,
each in its own CUDA stream, to quickly build a distribution of the sample mean.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h> // For random number generation on the GPU

const int ORIGINAL_DATA_SIZE = 1 << 16; // 65,536 data points
const int NUM_BOOTSTRAP_SAMPLES = 128;   // Number of bootstrap samples to generate
const int THREADS_PER_BLOCK = 256;

// Kernel to calculate the mean of one bootstrap sample
__global__ void bootstrap_mean_kernel(
    const float* original_data,
    float* bootstrap_means,
    int data_size,
    int sample_idx,
    unsigned long long seed)
{
    // Use a temporary array in shared memory for reduction
    extern __shared__ float sdata[];

    // Each thread needs its own random state
    curandState_t rand_state;
    curand_init(seed, sample_idx, threadIdx.x, &rand_state);

    // Each thread calculates a partial sum
    float my_sum = 0.0f;
    int grid_stride = gridDim.x * blockDim.x;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < data_size; i += grid_stride) {
        // Generate a random index to sample with replacement
        int rand_idx = curand(&rand_state) % data_size;
        my_sum += original_data[rand_idx];
    }
    sdata[threadIdx.x] = my_sum;
    __syncthreads();

    // Perform reduction in shared memory to get the sum for the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 of the block writes the final sum for this bootstrap sample
    if (threadIdx.x == 0) {
        bootstrap_means[sample_idx] = sdata[0] / data_size;
    }
}

int main() {
    float milli;
    cudaEvent_t start, stop;

    // --- Host Data Setup ---
    float* h_original_data = (float*)malloc(ORIGINAL_DATA_SIZE * sizeof(float));
    float* h_bootstrap_means = (float*)malloc(NUM_BOOTSTRAP_SAMPLES * sizeof(float));

    // Generate some sample data on the host (e.g., from a normal distribution)
    srand(time(NULL));
    for (int i = 0; i < ORIGINAL_DATA_SIZE; ++i) {
        h_original_data[i] = (float)rand() / RAND_MAX * 10.0f + 5.0f; // Random floats between 5 and 15
    }

    // --- Device Data Setup ---
    float* d_original_data;
    float* d_bootstrap_means;
    cudaMalloc(&d_original_data, ORIGINAL_DATA_SIZE * sizeof(float));
    cudaMalloc(&d_bootstrap_means, NUM_BOOTSTRAP_SAMPLES * sizeof(float));

    // Copy original data to the device
    cudaMemcpy(d_original_data, h_original_data, ORIGINAL_DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // --- Stream and Kernel Launch Setup ---
    cudaStream_t streams[NUM_BOOTSTRAP_SAMPLES];
    int num_blocks = (ORIGINAL_DATA_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    printf("Starting %d parallel bootstrap calculations...\n", NUM_BOOTSTRAP_SAMPLES);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < NUM_BOOTSTRAP_SAMPLES; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        
        // Launch a kernel for each bootstrap sample in its own stream
        bootstrap_mean_kernel<<<num_blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float), streams[i]>>>(
            d_original_data,
            d_bootstrap_means,
            ORIGINAL_DATA_SIZE,
            i,
            time(NULL) + i // Use a different seed for each sample
        );
    }

    // Wait for all streams to finish
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);

    printf("Finished bootstrap calculations.\n");
    printf("Execution Time: %f ms\n", milli);

    // Copy results back to host
    cudaMemcpy(h_bootstrap_means, d_bootstrap_means, NUM_BOOTSTRAP_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);

    // --- Analyze Results ---
    float total_mean = 0.0f;
    for (int i = 0; i < NUM_BOOTSTRAP_SAMPLES; ++i) {
        total_mean += h_bootstrap_means[i];
    }
    float final_bootstrap_mean = total_mean / NUM_BOOTSTRAP_SAMPLES;

    printf("Bootstrap Estimated Mean: %f\n", final_bootstrap_mean);
    
    // --- Cleanup ---
    free(h_original_data);
    free(h_bootstrap_means);
    cudaFree(d_original_data);
    cudaFree(d_bootstrap_means);
    for (int i = 0; i < NUM_BOOTSTRAP_SAMPLES; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaDeviceReset();

    return 0;
}
