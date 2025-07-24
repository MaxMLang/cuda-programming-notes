#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <algorithm> // For std::sort
#include <ctime>     // For time() function
#include "helper_cuda.h"

/*
 * @brief CUDA kernel to perform bootstrap resampling.
 *
 * Each thread calculates the mean of one bootstrap sample.
 * A bootstrap sample is created by drawing `sample_size` elements
 * from the `original_data` with replacement.
 *
 * @param original_data The initial dataset on the GPU.
 * @param data_size The total size of the original dataset.
 * @param bootstrap_means An array to store the calculated mean of each bootstrap sample.
 * @param resamples_for_this_launch The number of bootstrap resamples this specific kernel launch is responsible for.
 * @param sample_size The size of each bootstrap sample.
 * @param base_seed A seed for the random number generator.
 */
__global__ void bootstrap_kernel(const double *original_data, int data_size,
                                 double *bootstrap_means, int resamples_for_this_launch,
                                 int sample_size, unsigned long long base_seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard to ensure only the necessary number of threads do work for this launch.
    if (idx < resamples_for_this_launch) {
        // Initialize random number generator state for this thread
        curandState_t state;
        curand_init(base_seed, idx, 0, &state);

        double current_sum = 0.0;
        for (int i = 0; i < sample_size; ++i) {
            // Generate a random index to sample from the original data
            int random_idx = curand(&state) % data_size;
            current_sum += original_data[random_idx];
        }

        // Store the mean of this bootstrap sample.
        // The bootstrap_means pointer is already offset to the correct chunk for this stream.
        bootstrap_means[idx] = current_sum / sample_size;
    }
}

int main() {
    // --- Configuration ---
    int data_size = 1 << 20; // 1,048,576 data points
    int num_resamples = 1 << 14; // 16,384 bootstrap resamples
    int num_streams = 4;

    // --- Host Data Initialization ---
    double *h_original_data = (double*)malloc(data_size * sizeof(double));
    double *h_bootstrap_means = (double*)malloc(num_resamples * sizeof(double));

    // FIX: Seed the host random number generator for reproducibility.
    srand(42);
    // Generate some sample data (e.g., from a normal distribution)
    for (int i = 0; i < data_size; ++i) {
        h_original_data[i] = (double)rand() / RAND_MAX * 10.0;
    }

    // --- Device Data Allocation ---
    double *d_original_data, *d_bootstrap_means;
    checkCudaErrors(cudaMalloc(&d_original_data, data_size * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_bootstrap_means, num_resamples * sizeof(double)));

    // Copy original data to the device (this is done only once)
    checkCudaErrors(cudaMemcpy(d_original_data, h_original_data, data_size * sizeof(double), cudaMemcpyHostToDevice));

    // --- Streaming Execution ---
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    printf("Starting bootstrap with %d resamples across %d streams...\n", num_resamples, num_streams);
    float elapsedTime;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    int resamples_per_stream = num_resamples / num_streams;
    int block_size = 256;
    int grid_size = (resamples_per_stream + block_size - 1) / block_size;

    for (int i = 0; i < num_streams; ++i) {
        int offset = i * resamples_per_stream;

        // Launch one kernel per stream. Each kernel calculates a portion of the bootstrap means.
        // FIX: The arguments are now correct. The 4th argument tells the kernel how many
        // means to compute in this launch. The 5th argument is the size of each sample.
        bootstrap_kernel<<<grid_size, block_size, 0, streams[i]>>>(
            d_original_data,            // The full dataset
            data_size,                  // Size of the full dataset
            &d_bootstrap_means[offset], // Pointer to this stream's chunk of the output array
            resamples_per_stream,       // How many means to compute in this launch
            data_size,                  // The size of each bootstrap sample
            time(0) + i);               // A unique seed for this stream's RNG

        // Asynchronously copy the results for this stream back to the host.
        checkCudaErrors(cudaMemcpyAsync(&h_bootstrap_means[offset], &d_bootstrap_means[offset],
                                       resamples_per_stream * sizeof(double),
                                       cudaMemcpyDeviceToHost, streams[i]));
    }

    // FIX: Add an error check after launching all async operations
    checkCudaErrors(cudaGetLastError());

    // Wait for all streams to finish
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Bootstrap calculation time: %g ms\n", elapsedTime);

    // --- Calculate Confidence Interval ---
    // Sort the collected bootstrap means on the host
    std::sort(h_bootstrap_means, h_bootstrap_means + num_resamples);

    // Find the 2.5th and 97.5th percentiles for a 95% confidence interval
    int lower_idx = (int)(0.025 * num_resamples);
    int upper_idx = (int)(0.975 * num_resamples);

    printf("95%% Confidence Interval for the mean: [%.4f, %.4f]\n",
           h_bootstrap_means[lower_idx], h_bootstrap_means[upper_idx]);

    // --- Cleanup ---
    for (int i = 0; i < num_streams; ++i) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }
    free(h_original_data);
    free(h_bootstrap_means);
    checkCudaErrors(cudaFree(d_original_data));
    checkCudaErrors(cudaFree(d_bootstrap_means));

    return 0;
}
