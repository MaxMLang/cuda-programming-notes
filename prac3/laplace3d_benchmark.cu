//
// Program to solve and benchmark Laplace equation on a regular 3D grid
// Combines two different kernel implementations for comparison.
//
// To compile:
// nvcc laplace3d_benchmark.cu -o laplace3d_benchmark -arch=sm_70 --use_fast_math
//
// To run (for example, version 2 with 8x8x8 blocks):
// ./laplace3d_benchmark 2 8 8 8
//
// To run (for example, version 1 with 16x16 blocks):
// ./laplace3d_benchmark 1 16 16 0
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// KERNEL 1: Original version from laplace3d.cu
// Note: one thread per (i,j) pair, marching in the k-direction
////////////////////////////////////////////////////////////////////////
__global__ void GPU_laplace3d_v1(int NX, int NY, int NZ,
                              const float* __restrict__ d_u1,
                                    float* __restrict__ d_u2)
{
  int       i, j, k, IOFF, JOFF, KOFF;
  long long indg;
  float     u2, sixth=1.0f/6.0f;

  // Define global indices for the 2D grid of threads
  i    = threadIdx.x + blockIdx.x*blockDim.x;
  j    = threadIdx.y + blockIdx.y*blockDim.y;
  indg = i + j*NX;

  IOFF = 1;
  JOFF = NX;
  KOFF = NX*NY;

  // Check if the thread is within the x-y bounds
  if ( i < NX && j < NY ) {
    // Each thread iterates through the entire z-dimension
    for (k=0; k<NZ; k++) {
      if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
        u2 = d_u1[indg];  // Dirichlet b.c.'s
      }
      else {
        u2 = ( d_u1[indg-IOFF] + d_u1[indg+IOFF]
             + d_u1[indg-JOFF] + d_u1[indg+JOFF]
             + d_u1[indg-KOFF] + d_u1[indg+KOFF] ) * sixth;
      }
      d_u2[indg] = u2;

      // Move to the next point in the z-column
      indg += KOFF;
    }
  }
}


////////////////////////////////////////////////////////////////////////
// KERNEL 2: New version from laplace3d_new.cu
// Note: one thread per (i,j,k) grid point
////////////////////////////////////////////////////////////////////////
__global__ void GPU_laplace3d_v2(int NX, int NY, int NZ,
	         	      const float* __restrict__ d_u1,
			            float* __restrict__ d_u2)
{
  int       i, j, k, IOFF, JOFF, KOFF;
  long long indg;
  float     u2, sixth=1.0f/6.0f;

  // Define global indices for the 3D grid of threads
  i    = threadIdx.x + blockIdx.x*blockDim.x;
  j    = threadIdx.y + blockIdx.y*blockDim.y;
  k    = threadIdx.z + blockIdx.z*blockDim.z;

  IOFF = 1;
  JOFF = NX;
  KOFF = NX*NY;

  indg = i + j*JOFF + k*KOFF;

  // Check if the thread is within the 3D grid bounds
  if (i < NX && j < NY && k < NZ) {
    if (i==0 || i==NX-1 || j==0 || j==NY-1 || k==0 || k==NZ-1) {
      u2 = d_u1[indg];  // Dirichlet b.c.'s
    }
    else {
      u2 = ( d_u1[indg-IOFF] + d_u1[indg+IOFF]
           + d_u1[indg-JOFF] + d_u1[indg+JOFF]
           + d_u1[indg-KOFF] + d_u1[indg+KOFF] ) * sixth;
    }
    d_u2[indg] = u2;
  }
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv){

  // --- Benchmark Setup ---
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <version> <BLOCK_X> <BLOCK_Y> <BLOCK_Z>\n", argv[0]);
    fprintf(stderr, "  version: 1 for original, 2 for new\n");
    fprintf(stderr, "  For version 1, BLOCK_Z is ignored (use 0).\n");
    return 1;
  }

  int version = atoi(argv[1]);
  int BLOCK_X = atoi(argv[2]);
  int BLOCK_Y = atoi(argv[3]);
  int BLOCK_Z = atoi(argv[4]);

  int     NX=1024, NY=1024, NZ=1024, REPEAT=10;
  float  *d_u1, *d_u2, *d_foo;
  size_t  bytes = sizeof(float) * (size_t)NX*NY*NZ;

  printf("Grid: %d x %d x %d | Iterations: %d | Version: %d\n", NX, NY, NZ, REPEAT, version);
  if(version == 1) printf("Block Dims: %d x %d\n", BLOCK_X, BLOCK_Y);
  else printf("Block Dims: %d x %d x %d\n", BLOCK_X, BLOCK_Y, BLOCK_Z);

  // --- CUDA Initialization ---
  findCudaDevice(argc, argv);
  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // --- Memory Allocation and Initialization ---
  checkCudaErrors( cudaMalloc((void **)&d_u1, bytes) );
  checkCudaErrors( cudaMalloc((void **)&d_u2, bytes) );
  // In a real scenario, you'd initialize d_u1 with boundary conditions.
  // For a pure performance benchmark, we can skip initializing the data.
  cudaMemset(d_u1, 0, bytes);
  cudaMemset(d_u2, 0, bytes);


  // --- Kernel Execution ---
  cudaEventRecord(start);

  if (version == 1) {
    dim3 dimBlock(BLOCK_X, BLOCK_Y);
    dim3 dimGrid( (NX + BLOCK_X - 1) / BLOCK_X, (NY + BLOCK_Y - 1) / BLOCK_Y );
    for (int i=0; i<REPEAT; i++) {
        GPU_laplace3d_v1<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2);
        d_foo = d_u1; d_u1 = d_u2; d_u2 = d_foo;
    }
  } else if (version == 2) {
    dim3 dimBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 dimGrid( (NX + BLOCK_X - 1) / BLOCK_X, (NY + BLOCK_Y - 1) / BLOCK_Y, (NZ + BLOCK_Z - 1) / BLOCK_Z );
    for (int i=0; i<REPEAT; i++) {
        GPU_laplace3d_v2<<<dimGrid, dimBlock>>>(NX, NY, NZ, d_u1, d_u2);
        d_foo = d_u1; d_u1 = d_u2; d_u2 = d_foo;
    }
  } else {
      fprintf(stderr, "Error: version must be 1 or 2.\n");
      return 1;
  }
  
  // Check for any errors during kernel execution
  getLastCudaError("Kernel execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  
  printf("Total Time: %.2f (ms) | Time per iteration: %.3f (ms)\n\n", milli, milli/REPEAT);


  // --- Cleanup ---
  checkCudaErrors( cudaFree(d_u1) );
  checkCudaErrors( cudaFree(d_u2) );
  cudaDeviceReset();

  return 0;
}
