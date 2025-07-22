////////////////////////////////////////////////////////////////////////////////
//
// Program to solve Laplace equation and compute RMS change at each step.
// Fulfills Task 6 of the practical.
//
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// Laplace Kernel (one thread per grid point)
////////////////////////////////////////////////////////////////////////////////
__global__ void GPU_laplace3d(int NX, int NY, int NZ,
                  const float* __restrict__ d_u1,
                  float* __restrict__ d_u2)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z + blockIdx.z*blockDim.z;
  
  if (i >= NX || j >= NY || k >= NZ) return;

  long long indg = (long long)i + (long long)j*NX + (long long)k*NX*NY;

  if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
    d_u2[indg] = d_u1[indg]; // Dirichlet b.c.'s
  } else {
    float sixth = 1.0f/6.0f;
    d_u2[indg] = ( d_u1[indg-1] + d_u1[indg+1]
                 + d_u1[indg-NX] + d_u1[indg+NX]
                 + d_u1[indg-(long long)NX*NY] + d_u1[indg+(long long)NX*NY] ) * sixth;
  }
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: Compute Squared Change
// Computes (u2[i] - u1[i])^2 for each point and stores it in d_sq_change.
////////////////////////////////////////////////////////////////////////////////
__global__ void compute_squared_change(const float* d_u1, const float* d_u2, float* d_sq_change, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = d_u2[i] - d_u1[i];
        d_sq_change[i] = diff * diff;
    }
}

////////////////////////////////////////////////////////////////////////////////
// KERNEL: Reduction (Shuffle version from the other exercise)
// This is the same optimized reduction kernel needed for the sum.
////////////////////////////////////////////////////////////////////////////////
__global__ void reduction_shuffle_kernel(float* g_odata, const float* g_idata, int n)
{
    float my_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        my_sum += g_idata[i];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
    }
    extern __shared__ float temp[];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) temp[warp_id] = my_sum;
    __syncthreads();
    if (warp_id == 0) {
        my_sum = (lane_id < blockDim.x / 32) ? temp[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);
        }
        if (lane_id == 0) g_odata[blockIdx.x] = my_sum;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv){

  int NX=128, NY=128, NZ=128, REPEAT=50;
  int BLOCK_X=8, BLOCK_Y=8, BLOCK_Z=8;

  printf("Grid: %d x %d x %d | Iterations: %d | Block: %d x %d x %d\n", NX, NY, NZ, REPEAT, BLOCK_X, BLOCK_Y, BLOCK_Z);

  long long num_elements = (long long)NX*NY*NZ;
  size_t bytes = sizeof(float) * num_elements;
  
  float *h_u1, *d_u1, *d_u2, *d_foo;

  findCudaDevice(argc, argv);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // --- Memory Allocation ---
  h_u1 = (float *)malloc(bytes);
  checkCudaErrors( cudaMalloc((void **)&d_u1, bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_u2, bytes) );

  // --- Initialization ---
  for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
    long long ind = i + j*NX + (long long)k*NX*NY;
    h_u1[ind] = (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1) ? 1.0f : 0.0f;
  }
  checkCudaErrors( cudaMemcpy(d_u1, h_u1, bytes, cudaMemcpyHostToDevice) );
  cudaMemset(d_u2, 0, bytes); // Ensure d_u2 is zeroed out

  // --- Setup for RMS Calculation ---
  float *d_sq_change, *d_partial_sums, *h_partial_sums;
  int reduction_threads = 256;
  int reduction_blocks = (num_elements + reduction_threads - 1) / reduction_threads;
  checkCudaErrors( cudaMalloc((void**)&d_sq_change, bytes) );
  checkCudaErrors( cudaMalloc((void**)&d_partial_sums, sizeof(float) * reduction_blocks) );
  h_partial_sums = (float*) malloc(sizeof(float) * reduction_blocks);
  size_t reduction_shared_mem = (reduction_threads / 32) * sizeof(float);


  // --- Kernel Execution Loop ---
  dim3 dimGrid( (NX+BLOCK_X-1)/BLOCK_X, (NY+BLOCK_Y-1)/BLOCK_Y, (NZ+BLOCK_Z-1)/BLOCK_Z );
  dim3 dimBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);

  cudaEventRecord(start);

  for (int i=0; i<REPEAT; i++) {
    // 1. Perform one step of the Laplace solver
    GPU_laplace3d<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2);
    
    // 2. Compute the squared difference: (u2-u1)^2
    compute_squared_change<<<reduction_blocks, reduction_threads>>>(d_u1, d_u2, d_sq_change, num_elements);

    // 3. Reduce the squared differences to get the total sum
    reduction_shuffle_kernel<<<reduction_blocks, reduction_threads, reduction_shared_mem>>>(d_partial_sums, d_sq_change, num_elements);

    // 4. Swap pointers for the next iteration
    d_foo = d_u1; d_u1 = d_u2; d_u2 = d_foo;
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milli;
  cudaEventElapsedTime(&milli, start, stop);
  printf("Total GPU Time for %d iterations: %.2f ms\n", REPEAT, milli);

  // --- Final RMS Calculation ---
  // Copy the final partial sums back to host to compute the final RMS
  checkCudaErrors( cudaMemcpy(h_partial_sums, d_partial_sums, sizeof(float) * reduction_blocks, cudaMemcpyDeviceToHost) );
  
  float sum_sq_err = 0.0f;
  for (int i = 0; i < reduction_blocks; i++) {
      sum_sq_err += h_partial_sums[i];
  }
  float rms_change = sqrt(sum_sq_err / num_elements);
  printf("Final RMS change after %d iterations: %f\n", REPEAT, rms_change);
    
  // --- Cleanup ---
  free(h_u1);
  free(h_partial_sums);
  checkCudaErrors( cudaFree(d_u1) );
  checkCudaErrors( cudaFree(d_u2) );
  checkCudaErrors( cudaFree(d_sq_change) );
  checkCudaErrors( cudaFree(d_partial_sums) );
  cudaDeviceReset();
}