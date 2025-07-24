// trid_dyn.cu
//
// Program to perform Backward Euler time-marching on a 1D grid
// MODIFIED to use dynamic shared memory.
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// kernel function
////////////////////////////////////////////////////////////////////////

__global__ void GPU_trid_dyn(int NX, int niter, float *u)
{
  extern __shared__ float s_mem[];

  // Manually set up pointers for a, c, and d within the shared memory block
  float *a = s_mem;
  float *c = &s_mem[NX];
  float *d = &s_mem[2*NX];

  float aa, bb, cc, dd, bbi, lambda=1.0;
  int   tid;

  for (int iter=0; iter<niter; iter++) {

    // set tridiagonal coefficients and r.h.s.

    tid = threadIdx.x;
    bbi = 1.0f / (2.0f + lambda);
    
    if (tid>0)
      aa = -bbi;
    else
      aa = 0.0f;

    if (tid<blockDim.x-1)
      cc = -bbi;
    else
      cc = 0.0f;

    // On first iteration, u is global memory, otherwise use the previous d
    if (iter==0)
      dd = lambda*u[tid]*bbi;
    else
      dd = lambda*d[tid]*bbi; // Use previous iteration's result from shared memory

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

  u[tid] = d[tid];
}

////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void gold_trid(int, int, float*, float*);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

  int    NX = 256, niter = 10; // Increased NX to test dynamic shared memory

  float *h_u, *h_v, *h_c, *d_u;

  // initialise card
  findCudaDevice(argc, argv);

  // allocate memory on host and device
  h_u = (float *)malloc(sizeof(float)*NX);
  h_v = (float *)malloc(sizeof(float)*NX);
  h_c = (float *)malloc(sizeof(float)*NX);
  checkCudaErrors( cudaMalloc((void **)&d_u, sizeof(float)*NX) );

  // GPU execution
  for (int i=0; i<NX; i++) h_u[i] = 1.0f;
  checkCudaErrors( cudaMemcpy(d_u, h_u, sizeof(float)*NX, cudaMemcpyHostToDevice) );

  // ***MODIFICATION HERE***
  // Specify dynamic shared memory size in the kernel launch
  size_t shmem_size = 3 * NX * sizeof(float);
  GPU_trid_dyn<<<1, NX, shmem_size>>>(NX, niter, d_u);

  checkCudaErrors( cudaMemcpy(h_u, d_u, sizeof(float)*NX, cudaMemcpyDeviceToHost) );

  // CPU execution
  for (int i=0; i<NX; i++) h_v[i] = 1.0f;
  gold_trid(NX, niter, h_v, h_c);

  // Verification
  double err = 0.0, max_err = 0.0;
  for (int i=0; i<NX; i++) {
    err = fabs(h_u[i] - h_v[i]);
    if (err > max_err) max_err = err;
  }
  printf("Max error: %e\n", max_err);

  // Release GPU and CPU memory
  checkCudaErrors( cudaFree(d_u) );
  free(h_u);
  free(h_v);
  free(h_c);
}