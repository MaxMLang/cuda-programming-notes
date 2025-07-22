////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
// Benchmark version to test different memory access patterns.
//
// To compile:
// nvcc prac2_benchmark.cu -o prac2_benchmark -arch=sm_70
//
// To run (for example, Version 1 with 128 threads/block):
// ./prac2_benchmark 1 128
//
// To run (for example, Version 2 with 128 threads/block):
// ./prac2_benchmark 2 128
//
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int   N;
__constant__ float T, r, sigma, rho, alpha, dt, con1, con2;


////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////
__global__ void pathcalc(float *d_z, float *d_v, int version)
{
  float s1, s2, y1, y2, payoff;
  int   ind;
  int   thread_id_in_block = threadIdx.x;
  int   block_id = blockIdx.x;
  int   block_width = blockDim.x;

  // --- Index Calculation ---
  if (version == 1) {
    // Version 1: Coalesced Access
    // Data is laid out like: [Z1_path0, Z1_path1, ..., Z2_path0, Z2_path1, ...]
    ind = thread_id_in_block + 2 * N * block_id * block_width;
  } else {
    // Version 2: Strided (Non-Coalesced) Access
    // Data is laid out like: [Z1_path0, Z2_path0, Z1_path1, Z2_path1, ...]
    ind = 2 * N * thread_id_in_block + 2 * N * block_id * block_width;
  }

  // --- Path Calculation ---
  s1 = 1.0f;
  s2 = 1.0f;

  for (int n = 0; n < N; n++) {
    y1 = d_z[ind];

    if (version == 1) {
      ind += block_width; // Move to the next Z1 for this thread
    } else {
      ind += 1;           // Move to Z2 for this path
    }

    y2 = rho * y1 + alpha * d_z[ind];

    if (version == 1) {
      ind += block_width; // Move to the Z1 for the next timestep
    } else {
      ind += 1;           // Move to Z1 for the next timestep
    }

    s1 = s1 * (con1 + con2 * y1);
    s2 = s2 * (con1 + con2 * y2);
  }

  // --- Store Payoff ---
  payoff = 0.0f;
  if (fabs(s1 - 1.0f) < 0.1f && fabs(s2 - 1.0f) < 0.1f) {
    payoff = exp(-r * T);
  }
  d_v[thread_id_in_block + block_id * block_width] = payoff;
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv){
    
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <version> <threads_per_block>\n", argv[0]);
    fprintf(stderr, "  version: 1 for Coalesced, 2 for Strided\n");
    return 1;
  }

  int version = atoi(argv[1]);
  int threads_per_block = atoi(argv[2]);

  int     NPATH=9600000, h_N=100;
  float   h_T, h_r, h_sigma, h_rho, h_alpha, h_dt, h_con1, h_con2;
  float  *h_v, *d_v, *d_z;
  double  sum1, sum2;

  printf("Paths: %d | Timesteps: %d | Version: %d | Threads/Block: %d\n", NPATH, h_N, version, threads_per_block);

  findCudaDevice(argc, argv);

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  h_v = (float *)malloc(sizeof(float)*NPATH);
  checkCudaErrors( cudaMalloc((void **)&d_v, sizeof(float)*NPATH) );
  checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*2*h_N*NPATH) );

  h_T     = 1.0f;
  h_r     = 0.05f;
  h_sigma = 0.1f;
  h_rho   = 0.5f;
  h_alpha = sqrt(1.0f-h_rho*h_rho);
  h_dt    = 1.0f/h_N;
  h_con1  = 1.0f + h_r*h_dt;
  h_con2  = sqrt(h_dt)*h_sigma;

  checkCudaErrors( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
  checkCudaErrors( cudaMemcpyToSymbol(T,    &h_T,    sizeof(h_T)) );
  checkCudaErrors( cudaMemcpyToSymbol(r,    &h_r,    sizeof(h_r)) );
  checkCudaErrors( cudaMemcpyToSymbol(sigma,&h_sigma,sizeof(h_sigma)) );
  checkCudaErrors( cudaMemcpyToSymbol(rho,  &h_rho,  sizeof(h_rho)) );
  checkCudaErrors( cudaMemcpyToSymbol(alpha,&h_alpha,sizeof(h_alpha)) );
  checkCudaErrors( cudaMemcpyToSymbol(dt,   &h_dt,   sizeof(h_dt)) );
  checkCudaErrors( cudaMemcpyToSymbol(con1, &h_con1, sizeof(h_con1)) );
  checkCudaErrors( cudaMemcpyToSymbol(con2, &h_con2, sizeof(h_con2)) );

  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
  checkCudaErrors( curandGenerateNormal(gen, d_z, 2*h_N*NPATH, 0.0f, 1.0f) );

  int num_blocks = NPATH / threads_per_block;
  
  cudaEventRecord(start);
  pathcalc<<<num_blocks, threads_per_block>>>(d_z, d_v, version);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  getLastCudaError("pathcalc execution failed\n");
  printf("Monte Carlo kernel execution time (ms): %.3f \n", milli);

  checkCudaErrors( cudaMemcpy(h_v, d_v, sizeof(float)*NPATH, cudaMemcpyDeviceToHost) );

  sum1 = 0.0;
  sum2 = 0.0;
  for (int i=0; i<NPATH; i++) {
    sum1 += h_v[i];
    sum2 += h_v[i]*h_v[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
	 sum1/NPATH, sqrt((sum2/NPATH - (sum1/NPATH)*(sum1/NPATH))/NPATH) );

  checkCudaErrors( curandDestroyGenerator(gen) );
  free(h_v);
  checkCudaErrors( cudaFree(d_v) );
  checkCudaErrors( cudaFree(d_z) );
  cudaDeviceReset();
}
