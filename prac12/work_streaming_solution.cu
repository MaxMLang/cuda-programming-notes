#include <stdio.h>
#include <cuda.h>
#include <cmath> // For fabs and cos
#include "helper_cuda.h"

// The kernel remains the same, it just processes the data it's given.
__global__ void do_work(double *data, int N, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (i < N) {
        for (int j = 0; j < 20; j++) {
            data[i] = cos(data[i]);
            data[i] = sqrt(fabs(data[i]));
        }
    }
}

int main()
{
    // Total data size (approx 1GB)
    int total_data = 1 << 27;
    double *d_data;
    double *h_data;

    // STEP 1: Use pinned (page-locked) host memory for faster, asynchronous transfers.
    // This is a prerequisite for overlapping memory copies with kernel execution.
    checkCudaErrors(cudaMallocHost((void**)&h_data, total_data * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_data, total_data * sizeof(double)));

    // Initialise host data with random values
    srand(0);
    for (int i = 0; i < total_data; i++) {
        h_data[i] = (double)rand() / (double)RAND_MAX;
    }

    // STEP 2: Define the number of streams (chunks) to break the work into.
    int num_streams = 8;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }

    // Set up CUDA events for accurate timing
    float time;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    int blocksize = 256;
    // Calculate the size of each data chunk
    int chunk_size = total_data / num_streams;

    // STEP 3: Loop through the streams, issuing asynchronous operations.
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        
        // Calculate the number of blocks needed for this specific chunk
        int nblocks = (chunk_size - 1) / blocksize + 1;

        // Asynchronously copy a chunk of data from Host to Device in its specific stream.
        // The CPU does not wait for this to complete.
        checkCudaErrors(cudaMemcpyAsync(&d_data[offset], &h_data[offset],
                                       chunk_size * sizeof(double),
                                       cudaMemcpyHostToDevice, streams[i]));

        // Launch the kernel to process the data chunk in the same stream.
        // This kernel launch is queued after the copy in the same stream.
        // It will only execute after the HtoD copy for this stream is complete.
        do_work<<<nblocks, blocksize, 0, streams[i]>>>(d_data, total_data, offset);

        // Asynchronously copy the processed chunk from Device to Host in the same stream.
        // This is queued after the kernel. It will execute after the kernel for this stream is done.
        checkCudaErrors(cudaMemcpyAsync(&h_data[offset], &d_data[offset],
                                       chunk_size * sizeof(double),
                                       cudaMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all operations in all streams to complete before stopping the timer.
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
    printf("Total processing time with streaming: %g ms\n", time);

    // Clean up streams and memory
    for (int i = 0; i < num_streams; i++) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
    }
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFreeHost(h_data)); // Use cudaFreeHost for pinned memory

    return EXIT_SUCCESS;
}
