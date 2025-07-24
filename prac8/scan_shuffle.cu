#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cooperative_groups.h>

#include <helper_cuda.h>

namespace cg = cooperative_groups;

///////////////////////////////////////////////////////////////////////
// CPU routine for verification
///////////////////////////////////////////////////////////////////////

void scan_gold(float* odata, float* idata, const unsigned int len) 
{
  odata[0] = 0;
  for(unsigned int i=1; i<len; i++) odata[i] = idata[i-1] + odata[i-1];
}

///////////////////////////////////////////////////////////////////////
// GPU Kernel using Shuffle instructions
///////////////////////////////////////////////////////////////////////

__global__ void scan_shuffle(float *g_odata, float *g_idata, unsigned int n)
{
    // Shared memory for warp-level communication
    extern __shared__ float warp_sums[];

    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int l_tid = threadIdx.x;

    // Determine warp and lane IDs
    int warp_id = l_tid / warpSize;
    int lane_id = l_tid % warpSize;

    // Load data into a register
    float val = (g_tid < n) ? g_idata[g_tid] : 0.0f;

    // Get the current thread block's warp
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());

    // --- Step 1: Intra-warp scan using shuffle ---
    // The mask 0xffffffff means all threads in the warp participate
    for (int d = 1; d < warpSize; d *= 2) {
        float temp = warp.shfl_up(val, d);
        if (lane_id >= d) {
            val += temp;
        }
    }

    // --- Step 2: Inter-warp scan using shared memory ---
    // The last lane in each warp writes its total sum to shared memory
    if (lane_id == warpSize - 1) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // The first warp performs a scan on the warp sums in shared memory
    if (warp_id == 0) {
        // This is a sequential scan, but it's on a small amount of data (num_warps)
        float warp_scan = warp_sums[lane_id];
        for (int d = 1; d < blockDim.x / warpSize; d *= 2) {
            float temp = (lane_id >= d) ? warp_sums[lane_id - d] : 0.0f;
            if (lane_id >= d) {
                warp_scan += temp;
            }
        }
        warp_sums[lane_id] = warp_scan;
    }
    __syncthreads();

    // --- Step 3: Add the scanned warp sums back to each element ---
    // Each thread (except those in the first warp) adds the cumulative sum
    // from the preceding warps to its own value.
    if (warp_id > 0) {
        val += warp_sums[warp_id - 1];
    }
    
    // --- Step 4: Write exclusive scan result to global memory ---
    // The final value for g_odata[g_tid] is the value from the previous thread
    // within the entire block. We can get this with one final shuffle.
    // We use cg::shfl_up to get the value from the thread with lane_id-1
    // and block-wide communication for the warp boundaries.
    float final_val = 0.0f;
    if (g_tid > 0) {
        // To get the previous element's value, we can read from g_idata and do the scan again
        // or more efficiently, we can use the values we've computed.
        // The value `val` currently holds the inclusive scan result.
        // The exclusive scan result is the inclusive scan of the previous element.
        // This is tricky to get across warps efficiently without another sync.
        // A simpler way is to write the inclusive scan and then have another kernel shift it.
        // For this example, we will just write the inclusive scan result for simplicity.
        // The gold standard can be adjusted to check against an inclusive scan.
        final_val = val;
    }

    // For an exclusive scan, we need the value from the thread before us.
    // This is simple: the final result for thread `i` is `val` from thread `i-1`.
    // We can read the original data again and compute. For now, let's stick to the prompt.
    // The prompt asks for an output which is the sum of *preceding* elements.
    if (g_tid < n) {
       g_odata[g_tid] = val - g_idata[g_tid]; // Inclusive to Exclusive conversion
    }
}


////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_threads, num_elements, mem_size, shared_mem_size;

  findCudaDevice(argc, argv);

  num_threads  = 512;
  num_elements = num_threads; // Keep it to one block for this example
  mem_size     = sizeof(float) * num_elements;

  // Shared memory for warp sums. One float per warp.
  shared_mem_size = sizeof(float) * (num_threads / 32);

  float *h_data, *h_result, *reference;
  float *d_idata, *d_odata;

  h_data = (float*) malloc(mem_size);
  h_result = (float*) malloc(mem_size);
  reference = (float*) malloc(mem_size);
      
  for(int i=0; i<num_elements; i++) 
    h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));

  scan_gold(reference, h_data, num_elements);

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, mem_size) );
  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice));

  scan_shuffle<<<1, num_threads, shared_mem_size>>>(d_odata, d_idata, num_elements);
  getLastCudaError("scan_shuffle kernel execution failed");

  checkCudaErrors( cudaMemcpy(h_result, d_odata, mem_size, cudaMemcpyDeviceToHost) );

  float err=0.0;
  for (int i=0; i<num_elements; i++)
    err += (h_result[i] - reference[i])*(h_result[i] - reference[i]);
  printf("RMS scan error  = %f\n",sqrt(err/num_elements));

  free(h_data);
  free(h_result);
  free(reference);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  cudaDeviceReset();
}
