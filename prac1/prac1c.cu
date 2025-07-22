//
// include files
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <helper_cuda.h>

//
// MODIFIED kernel routine to accept three vectors for addition
//
__global__ void vector_add_kernel(float *a, float *b, float *c)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  c[tid] = a[tid] + b[tid];
}

//
// main code
//
int main(int argc, const char **argv)
{
  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;
  int   nblocks, nthreads, nsize, n;

  // initialise card
  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block
  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads;

  // allocate memory for arrays on the host
  h_a = (float *)malloc(nsize*sizeof(float));
  h_b = (float *)malloc(nsize*sizeof(float));
  h_c = (float *)malloc(nsize*sizeof(float));

  for (n=0; n<nsize; n++) {
    h_a[n] = (float)n;
    h_b[n] = (float)(2*n);
  }

  checkCudaErrors(cudaMalloc((void **)&d_a, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_b, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_c, nsize*sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_a, h_a, nsize*sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b, h_b, nsize*sizeof(float), cudaMemcpyHostToDevice));

  vector_add_kernel<<<nblocks,nthreads>>>(d_a, d_b, d_c);
  getLastCudaError("vector_add_kernel execution failed\n");

  checkCudaErrors(cudaMemcpy(h_c, d_c, nsize*sizeof(float), cudaMemcpyDeviceToHost));
  for (n=0; n<nsize; n++) {
    printf("n=%d, a=%f, b=%f, c=%f \n", n, h_a[n], h_b[n], h_c[n]);
  }

  checkCudaErrors(cudaFree(d_a));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_c));

  free(h_a);
  free(h_b);
  free(h_c);

  // CUDA exit
  cudaDeviceReset();

  return 0;
}