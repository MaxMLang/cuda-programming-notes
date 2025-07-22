////////////////////////////////////////////////////////////////////////////////
//
// Practical 4 -- Completed Reduction Implementation (Corrected with double)
//
// This version uses 'double' to maintain precision for large sums.
//
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// CPU routine for verification
////////////////////////////////////////////////////////////////////////////////
double reduction_gold(double* idata, int len)
{
  double sum = 0.0;
  for(int i = 0; i < len; i++) sum += idata[i];
  return sum;
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: Optimized Multi-Block Reduction with Shuffle Instructions (double)
////////////////////////////////////////////////////////////////////////////////
__global__ void reduction_shuffle_kernel(double* g_odata, const double* g_idata, int n)
{
    double my_sum = 0.0;
    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        my_sum += g_idata[i];
    }

    // --- Warp-level reduction using shuffle instructions ---
    for (int offset = 16; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
    }

    // --- Block-level reduction ---
    extern __shared__ double temp[];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        temp[warp_id] = my_sum;
    }
    __syncthreads();

    // The first warp (threads 0-31) sums the results from the warp leaders
    if (warp_id == 0) {
        my_sum = (lane_id < blockDim.x / 32) ? temp[lane_id] : 0.0;
        for (int offset = 16; offset > 0; offset /= 2) {
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }
        if (lane_id == 0) {
            g_odata[blockIdx.x] = my_sum;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, const char** argv)
{
    int num_elements = 1 << 24; // ~16.7 million elements
    int num_threads = 256;       // Threads per block
    int num_blocks = (num_elements + num_threads - 1) / num_threads;

    size_t mem_size = sizeof(double) * num_elements;
    size_t partial_sum_mem_size = sizeof(double) * num_blocks;

    double *h_data, *d_idata, *d_odata, *h_partial_sums;

    findCudaDevice(argc, argv);

    // Allocate host memory
    h_data = (double*) malloc(mem_size);
    h_partial_sums = (double*) malloc(partial_sum_mem_size);
    for(int i = 0; i < num_elements; i++)
        h_data[i] = floor(10.0*(rand()/(double)RAND_MAX));

    // Compute reference solution
    printf("Computing gold standard on CPU...\n");
    double sum_gold = reduction_gold(h_data, num_elements);
    printf("Expected Sum = %f\n\n", sum_gold);

    // Allocate device memory
    checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
    checkCudaErrors( cudaMalloc((void**)&d_odata, partial_sum_mem_size) );

    // Copy host memory to device
    checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );

    // --- Execute the Multi-Block Shuffle Kernel ---
    printf("Running Multi-Block Shuffle Reduction...\n");
    printf("Grid size: %d blocks, %d threads per block\n", num_blocks, num_threads);

    size_t shared_mem_size = (num_threads / 32) * sizeof(double); // Shared mem for warp results

    reduction_shuffle_kernel<<<num_blocks, num_threads, shared_mem_size>>>(d_odata, d_idata, num_elements);
    getLastCudaError("reduction_shuffle_kernel execution failed");

    // Copy partial sums from device to host
    checkCudaErrors( cudaMemcpy(h_partial_sums, d_odata, partial_sum_mem_size, cudaMemcpyDeviceToHost) );

    // Final reduction on the host
    double sum_gpu = 0.0;
    for(int i = 0; i < num_blocks; i++) {
        sum_gpu += h_partial_sums[i];
    }

    // Check results
    printf("GPU Sum         = %f\n", sum_gpu);
    printf("Reduction Error = %f\n", sum_gpu - sum_gold);

    // Cleanup
    free(h_data);
    free(h_partial_sums);
    checkCudaErrors( cudaFree(d_idata) );
    checkCudaErrors( cudaFree(d_odata) );
    cudaDeviceReset();
}