// trid_multi.cu
//
// Solves M independent tridiagonal systems in parallel using M blocks.
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// kernel function
////////////////////////////////////////////////////////////////////////
__global__ void GPU_trid_multi(int NX, int M, int niter, float *u)
{
  extern __shared__ float s_mem[];

  // Manually set up pointers for a, c, and d within the shared memory block
  float *a = s_mem;
  float *c = &s_mem[NX];
  float *d = &s_mem[2*NX];

  float aa, bb, cc, dd, bbi, lambda=1.0;
  
  // Each thread identifies its local index and its block (problem) index
  int tid = threadIdx.x;
  int block_id = blockIdx.x;
  int global_idx_start = block_id * NX;

  for (int iter=0; iter<niter; iter++) {

    // set tridiagonal coefficients and r.h.s.
    bbi = 1.0f / (2.0f + lambda);
    
    if (tid > 0)
      aa = -bbi;
    else
      aa = 0.0f;

    if (tid < blockDim.x-1)
      cc = -bbi;
    else
      cc = 0.0f;
    
    // On first iteration, u is global memory, otherwise use the previous d
    if (iter==0)
      dd = lambda * u[global_idx_start + tid] * bbi;
    else
      dd = lambda * d[tid] * bbi; // Use previous iteration's result from shared memory

    a[tid] = aa;
    c[tid] = cc;
    d[tid] = dd;

    // parallel cyclic reduction
    for (int nt=1; nt<NX; nt=2*nt) {
      __syncthreads();  // finish writes before reads

      bb = 1.0f;

      if (tid-nt >= 0) {
        dd = dd - a[tid]*d[tid-nt];
        bb = bb - a[tid]*c[tid-nt];
        aa =    - a[tid]*a[tid-nt];
      }

      if (tid+nt < NX) {
        dd = dd - c[tid]*d[tid+nt];
        bb = bb - c[tid]*a[tid+nt];
        cc =    - c[tid]*c[tid+nt];
      }

      __syncthreads();  // finish reads before writes

      bbi = 1.0f / bb;
      aa  = aa*bbi;
      cc  = cc*bbi;
      dd  = dd*bbi;

      a[tid] = aa;
      c[tid] = cc;
      d[tid] = dd;
    }
  }

  u[global_idx_start + tid] = d[tid];
}

////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////
void gold_trid_multi(int, int, int, float*, float*);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv){

  int NX = 128, M = 64, niter = 10;
  size_t total_size = M * NX;

  float *h_u, *h_v, *h_c, *d_u;

  findCudaDevice(argc, argv);

  // Allocate memory for M systems
  h_u = (float *)malloc(sizeof(float)*total_size);
  h_v = (float *)malloc(sizeof(float)*total_size);
  h_c = (float *)malloc(sizeof(float)*total_size);
  checkCudaErrors( cudaMalloc((void **)&d_u, sizeof(float)*total_size) );

  // GPU execution
  for (int j=0; j<M; j++) {
      for (int i=0; i<NX; i++) {
          // Initialize each system with a different starting value
          h_u[j*NX + i] = 1.0f * (j+1); 
      }
  }
  checkCudaErrors( cudaMemcpy(d_u, h_u, sizeof(float)*total_size, cudaMemcpyHostToDevice) );

  size_t shmem_size = 3 * NX * sizeof(float);
  GPU_trid_multi<<<M, NX, shmem_size>>>(NX, M, niter, d_u);

  checkCudaErrors( cudaMemcpy(h_u, d_u, sizeof(float)*total_size, cudaMemcpyDeviceToHost) );

  // CPU execution
  for (int j=0; j<M; j++) {
      for (int i=0; i<NX; i++) {
          h_v[j*NX + i] = 1.0f * (j+1);
      }
  }
  gold_trid_multi(NX, M, niter, h_v, h_c);

  // Verification
  double err = 0.0, max_err = 0.0;
  for (size_t i=0; i < total_size; i++) {
    err = fabs(h_u[i] - h_v[i]);
    if (err > max_err) max_err = err;
  }
  printf("Max error across all %d systems: %e\n", M, max_err);

  // Release memory
  checkCudaErrors( cudaFree(d_u) );
  free(h_u);
  free(h_v);
  free(h_c);
}