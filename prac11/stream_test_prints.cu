/*
This is based on an example developed by Mark Harris for his NVIDIA blog:
http://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
-- I have added some timing and print statements for debugging.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>

const int N = 1 << 20;

__global__ void kernel(float *x, int n, int kernel_id)
{
    // Only thread 0 in the block prints the message
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Kernel %d: starts.\n", kernel_id);
    }

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }

    // Synchronize all threads in the block before printing the finish message
    // to ensure the computation is done.
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Kernel %d: finishes.\n", kernel_id);
    }
}

int main()
{
    // initialise CUDA timing, and start timer
    float milli;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // set up 8 streams
    const int num_streams = 8;
    cudaStream_t streams[num_streams];
    float *data[num_streams];

    // loop over 8 streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&data[i], N * sizeof(float));
        
        // launch one worker kernel per stream, passing the loop index as a kernel_id
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N, i);
    }

    // wait for completion of all kernels
    cudaDeviceSynchronize();

    // stop timer and report execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    printf("Execution time with prints (ms): %f \n", milli);

    // Clean up streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(data[i]);
    }

    cudaDeviceReset();

    return 0;
}
