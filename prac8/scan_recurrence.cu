#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

// Define a structure for our recurrence coefficients (a, b)
// We use float2 which is a native CUDA vector type.
typedef float2 RecurrencePair;

///////////////////////////////////////////////////////////////////////
// CPU routine for verification
///////////////////////////////////////////////////////////////////////

void recurrence_gold(RecurrencePair* odata, RecurrencePair* idata, const unsigned int len) 
{
  // The 'scan' on a recurrence gives the coefficients to get from y_0 to y_n
  // odata[n] = (A_n, B_n) such that y_n = A_n * y_0 + B_n
  odata[0] = make_float2(1.0f, 0.0f); // Identity element: y_0 = 1*y_0 + 0

  for(unsigned int i = 1; i < len; i++) {
    // Combine odata[i-1] = (a_prev, b_prev) with idata[i-1] = (a_curr, b_curr)
    RecurrencePair prev_coeff = odata[i-1];
    RecurrencePair curr_coeff = idata[i-1];

    // New coefficients (a_new, b_new) are:
    // a_new = a_curr * a_prev
    // b_new = a_curr * b_prev + b_curr
    odata[i].x = curr_coeff.x * prev_coeff.x;
    odata[i].y = curr_coeff.x * prev_coeff.y + curr_coeff.y;
  }
}

///////////////////////////////////////////////////////////////////////
// GPU Kernel for Recurrence Scan
///////////////////////////////////////////////////////////////////////

// This is the custom operator for our recurrence relation
__device__ inline RecurrencePair recurrence_op(RecurrencePair p1, RecurrencePair p2) {
    RecurrencePair res;
    res.x = p2.x * p1.x;
    res.y = p2.x * p1.y + p2.y;
    return res;
}

__global__ void scan_recurrence(RecurrencePair *g_odata, RecurrencePair *g_idata)
{
  extern __shared__ RecurrencePair temp_s[];

  int tid = threadIdx.x;
  RecurrencePair temp_val;

  // Load input into shared memory
  temp_val = g_idata[tid];
  temp_s[tid] = temp_val;

  // Perform scan using the custom recurrence operator
  for (int d = 1; d < blockDim.x; d *= 2) {
    __syncthreads();
    if (tid >= d) {
        RecurrencePair prev_val = temp_s[tid - d];
        temp_val = recurrence_op(prev_val, temp_val);
    }
    __syncthreads();
    temp_s[tid] = temp_val;
  }

  __syncthreads();

  // Write exclusive scan result to global memory
  RecurrencePair final_val = make_float2(1.0f, 0.0f); // Identity element
  if (tid > 0) {
    final_val = temp_s[tid - 1];
  }
  g_odata[tid] = final_val;
}


////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_threads, num_elements;
  size_t mem_size, shared_mem_size;

  findCudaDevice(argc, argv);

  num_threads  = 512;
  num_elements = num_threads; // One block for simplicity
  mem_size     = sizeof(RecurrencePair) * num_elements;

  RecurrencePair *h_data, *h_result, *reference;
  RecurrencePair *d_idata, *d_odata;

  // Allocate host memory
  h_data = (RecurrencePair*) malloc(mem_size);
  h_result = (RecurrencePair*) malloc(mem_size);
  reference = (RecurrencePair*) malloc(mem_size);
      
  // Initialize with some data
  for(int i=0; i<num_elements; i++) {
    h_data[i].x = 1.01f; // 'a' coefficient (e.g., interest rate)
    h_data[i].y = 10.0f; // 'b' coefficient (e.g., monthly deposit)
  }

  // Compute reference solution
  recurrence_gold(reference, h_data, num_elements);

  // Allocate device memory
  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, mem_size) );
  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice));

  // Execute the kernel
  shared_mem_size = sizeof(RecurrencePair) * num_threads;
  scan_recurrence<<<1, num_threads, shared_mem_size>>>(d_odata, d_idata);
  getLastCudaError("scan_recurrence kernel execution failed");

  checkCudaErrors( cudaMemcpy(h_result, d_odata, mem_size, cudaMemcpyDeviceToHost) );

  // Check results
  float err_a = 0.0, err_b = 0.0;
  for (int i=0; i<num_elements; i++) {
    err_a += (h_result[i].x - reference[i].x) * (h_result[i].x - reference[i].x);
    err_b += (h_result[i].y - reference[i].y) * (h_result[i].y - reference[i].y);
  }
  printf("RMS error (a) = %f\n", sqrt(err_a / num_elements));
  printf("RMS error (b) = %f\n", sqrt(err_b / num_elements));

  free(h_data);
  free(h_result);
  free(reference);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  cudaDeviceReset();
}
