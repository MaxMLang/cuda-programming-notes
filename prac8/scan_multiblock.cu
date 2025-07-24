#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

///////////////////////////////////////////////////////////////////////
// CPU routine for verification
///////////////////////////////////////////////////////////////////////

void scan_gold(float* odata, float* idata, const unsigned int len) 
{
  odata[0] = 0;
  for(unsigned int i=1; i<len; i++) odata[i] = idata[i-1] + odata[i-1];
}

///////////////////////////////////////////////////////////////////////
// GPU Kernels for Multi-Block Scan
///////////////////////////////////////////////////////////////////////

/**
 * @brief Kernel 1: Performs a scan within each block and writes the block's total sum
 * to a separate array (d_block_sums). The scan result within the block is also written
 * back to the output array. This is an intermediate result.
 */
__global__ void block_scan(float *g_odata, float *g_idata, float *d_block_sums, unsigned int n)
{
  // Dynamically allocated shared memory
  extern __shared__  float temp[];

  int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
  int l_tid = threadIdx.x;

  // Load input into shared memory
  // Handle elements beyond the array size by loading 0.0
  if (g_tid < n)
    temp[l_tid] = g_idata[g_tid];
  else
    temp[l_tid] = 0.0f;
  
  __syncthreads();

  // Parallel scan (up-sweep and down-sweep phases can be more efficient,
  // but this simpler version is easier to understand)
  for (int d = 1; d < blockDim.x; d *= 2) {
    __syncthreads();
    float val = (l_tid >= d) ? temp[l_tid - d] : 0.0f;
    __syncthreads();
    if (l_tid >= d) temp[l_tid] += val;
  }
  
  __syncthreads();

  // Write intermediate result to global memory
  if (g_tid < n)
    g_odata[g_tid] = (l_tid > 0) ? temp[l_tid-1] : 0.0f;
  
  // Last thread in each block writes the block's total sum
  if (l_tid == blockDim.x - 1) {
    d_block_sums[blockIdx.x] = temp[l_tid];
  }
}

/**
 * @brief Kernel 2: A simple kernel to add the scanned block sums back to the
 * intermediate results, producing the final correct scan.
 */
__global__ void add_block_sums(float *g_odata, float *d_block_sums, unsigned int n)
{
  int g_tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (g_tid < n) {
      g_odata[g_tid] += d_block_sums[blockIdx.x];
  }
}


////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  unsigned int num_threads, num_elements, num_blocks;

  // Use a larger dataset to demonstrate multi-block capability
  num_elements = 1000000;
  num_threads  = 512;
  num_blocks   = (num_elements + num_threads - 1) / num_threads;

  printf("Number of elements: %u\n", num_elements);
  printf("Threads per block:  %u\n", num_threads);
  printf("Number of blocks:   %u\n\n", num_blocks);
  
  unsigned int mem_size = sizeof(float) * num_elements;
  unsigned int block_sum_mem_size = sizeof(float) * num_blocks;
  unsigned int shared_mem_size = sizeof(float) * num_threads;


  float *h_data, *h_result, *reference;
  float *d_idata, *d_odata, *d_block_sums;

  // Initialise card
  findCudaDevice(argc, argv);

  // Allocate host memory
  h_data = (float*) malloc(mem_size);
  h_result = (float*) malloc(mem_size);
  reference = (float*) malloc(mem_size);
      
  for(unsigned int i=0; i<num_elements; i++) 
    h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));

  // Compute reference solution on CPU
  scan_gold(reference, h_data, num_elements);

  // Allocate device memory
  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_block_sums, block_sum_mem_size) );

  // Copy host memory to device input array
  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice));

  // --- KERNEL LAUNCH 1: Perform scan within each block ---
  block_scan<<<num_blocks, num_threads, shared_mem_size>>>(d_odata, d_idata, d_block_sums, num_elements);
  getLastCudaError("block_scan kernel execution failed");

  // --- KERNEL LAUNCH 2: Perform scan on the block sums ---
  // For simplicity, we reuse the block_scan kernel. For very large numbers of blocks,
  // this step would need to be done recursively.
  // We need a temporary array for the output of this second scan.
  float* d_scanned_block_sums;
  checkCudaErrors(cudaMalloc((void**)&d_scanned_block_sums, block_sum_mem_size));
  
  // The number of blocks from the first pass becomes the number of elements for this pass
  unsigned int num_elements_pass2 = num_blocks;
  unsigned int num_blocks_pass2 = (num_elements_pass2 + num_threads - 1) / num_threads;

  block_scan<<<num_blocks_pass2, num_threads, shared_mem_size>>>(d_scanned_block_sums, d_block_sums, d_block_sums, num_elements_pass2);
  getLastCudaError("scan of block_sums kernel execution failed");

  // --- KERNEL LAUNCH 3: Add the scanned block sums back ---
  add_block_sums<<<num_blocks, num_threads>>>(d_odata, d_scanned_block_sums, num_elements);
  getLastCudaError("add_block_sums kernel execution failed");

  // Copy result from device to host
  checkCudaErrors( cudaMemcpy(h_result, d_odata, mem_size, cudaMemcpyDeviceToHost) );

  // Check results
  float err=0.0;
  for (unsigned int i=0; i<num_elements; i++)
    err += (h_result[i] - reference[i])*(h_result[i] - reference[i]);
  printf("RMS scan error  = %f\n",sqrt(err/num_elements));

  // Cleanup memory
  free(h_data);
  free(h_result);
  free(reference);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );
  checkCudaErrors( cudaFree(d_block_sums) );
  checkCudaErrors( cudaFree(d_scanned_block_sums) );

  cudaDeviceReset();
}
