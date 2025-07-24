/*
This example demonstrates overlapped data transfer and computation using CUDA streams.
It simulates a scenario where chunks of data are continuously processed.
- Stream 0: Processes the current data chunk (n).
- Stream 1: Copies results from the previous chunk (n-1) back to the CPU.
- Stream 2: Copies input for the next chunk (n+1) to the GPU.
This requires the `--default-stream per-thread` flag to work correctly.
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int CHUNK_SIZE = 1 << 22; // Size of each data chunk (4M floats)
const int NUM_CHUNKS = 10;      // Number of chunks to process

// A simple kernel to perform some work on the data
__global__ void process_data(float *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        data[i] = sin(data[i]) * cos(data[i]);
    }
}

int main() {
    float milli;
    cudaEvent_t start, stop;

    // Host data buffers (pinned memory for faster async transfers)
    float *h_in[2], *h_out[2];
    // Device data buffers
    float *d_data[2];

    // Streams for different tasks
    cudaStream_t compute_stream, transfer_stream_up, transfer_stream_down;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&transfer_stream_up);   // Host to Device
    cudaStreamCreate(&transfer_stream_down); // Device to Host

    // Allocate pinned host memory and device memory
    for (int i = 0; i < 2; ++i) {
        cudaHostAlloc(&h_in[i], CHUNK_SIZE * sizeof(float), cudaHostAllocDefault);
        cudaHostAlloc(&h_out[i], CHUNK_SIZE * sizeof(float), cudaHostAllocDefault);
        cudaMalloc(&d_data[i], CHUNK_SIZE * sizeof(float));
        // Initialize input data
        for (int j = 0; j < CHUNK_SIZE; ++j) {
            h_in[i][j] = (float)(i * CHUNK_SIZE + j);
        }
    }

    printf("Starting overlapped processing of %d chunks...\n", NUM_CHUNKS);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- Main Processing Loop ---
    // Prime the pipeline: copy the first chunk to the device
    cudaMemcpyAsync(d_data[0], h_in[0], CHUNK_SIZE * sizeof(float), cudaMemcpyHostToDevice, transfer_stream_up);
    cudaStreamSynchronize(transfer_stream_up);

    for (int n = 0; n < NUM_CHUNKS; ++n) {
        int current_buf = n % 2;
        int next_buf = (n + 1) % 2;

        // 1. Launch compute kernel for the current chunk (n)
        process_data<<<1024, 256, 0, compute_stream>>>(d_data[current_buf], CHUNK_SIZE);

        // 2. Copy results from the previous chunk (n-1) back to host
        if (n > 0) {
            cudaMemcpyAsync(h_out[next_buf], d_data[next_buf], CHUNK_SIZE * sizeof(float), cudaMemcpyDeviceToHost, transfer_stream_down);
        }

        // 3. Copy input for the next chunk (n+1) to the device
        if (n < NUM_CHUNKS - 1) {
            cudaMemcpyAsync(d_data[next_buf], h_in[next_buf], CHUNK_SIZE * sizeof(float), cudaMemcpyHostToDevice, transfer_stream_up);
        }
        
        // To ensure correctness for this simple example, we can synchronize streams
        // In a real application, you might use events for finer-grained dependency management.
        cudaStreamSynchronize(compute_stream);
        if (n > 0) cudaStreamSynchronize(transfer_stream_down);
        if (n < NUM_CHUNKS - 1) cudaStreamSynchronize(transfer_stream_up);
    }

    cudaDeviceSynchronize(); // Wait for all operations to complete
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);

    printf("Finished processing.\n");
    printf("Total execution time: %f ms\n", milli);

    // Cleanup
    for (int i = 0; i < 2; ++i) {
        cudaFreeHost(h_in[i]);
        cudaFreeHost(h_out[i]);
        cudaFree(d_data[i]);
    }
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(transfer_stream_up);
    cudaStreamDestroy(transfer_stream_down);
    cudaDeviceReset();

    return 0;
}
