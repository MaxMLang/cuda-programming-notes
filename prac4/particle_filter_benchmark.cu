////////////////////////////////////////////////////////////////////////////////
//
// A 1D Particle Filter Benchmark in CUDA
//
// This program benchmarks two different parallel scan implementations for the
// resampling step of a particle filter. This version fixes a kernel timeout
// issue in the resampling step by using a parallel binary search.
//
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_cuda.h>

#define NUM_PARTICLES (1 << 20) // ~1 million particles
#define NUM_BLOCKS 256
#define NUM_THREADS 256
#define WARP_SIZE 32

// Particle data structure
typedef struct {
    double x;      // 1D position of the particle
    double weight;
} Particle;


////////////////////////////////////////////////////////////////////////////////
// UTILITY KERNELS
////////////////////////////////////////////////////////////////////////////////
__global__ void setup_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void extract_weights_kernel(const Particle* p, double* w, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        w[idx] = p[idx].weight;
    }
}

__global__ void normalize_kernel(double* cdf, double total, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && total > 0.0) {
        cdf[idx] /= total;
    }
}

// ADD THIS MISSING KERNEL
__global__ void add_block_sums_kernel(double* d_data, const double* d_block_sums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && blockIdx.x > 0) {
        d_data[idx] += d_block_sums[blockIdx.x];
    }
}


////////////////////////////////////////////////////////////////////////////////
// KERNEL 1: Prediction & KERNEL 2: Weighting (Unchanged)
////////////////////////////////////////////////////////////////////////////////
__global__ void predict_kernel(Particle* particles, curandState* states, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curandState localState = states[idx];
    double noise = curand_normal_double(&localState) * 0.1;
    particles[idx].x += noise;
    states[idx] = localState;
}

__global__ void weight_kernel(Particle* particles, double measurement, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double diff = particles[idx].x - measurement;
    double sigma = 0.5;
    particles[idx].weight = exp(-0.5 * diff * diff / (sigma * sigma));
}

////////////////////////////////////////////////////////////////////////////////
// SCAN VERSION 1: Blelloch Scan (Shared Memory)
////////////////////////////////////////////////////////////////////////////////
__global__ void scan_blelloch_kernel(double* d_out, const double* d_in, double* d_block_sums, int n) {
    extern __shared__ double temp[];
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x * 2;
    int idx1 = block_offset + tid;
    int idx2 = block_offset + tid + blockDim.x;

    temp[tid] = (idx1 < n) ? d_in[idx1] : 0.0;
    temp[tid + blockDim.x] = (idx2 < n) ? d_in[idx2] : 0.0;
    __syncthreads();

    for (int d = 1; d < blockDim.x * 2; d *= 2) {
        int i = 2 * d * (tid + 1) - 1;
        if (i < blockDim.x * 2) temp[i] += temp[i - d];
        __syncthreads();
    }

    if (tid == 0) {
        d_block_sums[blockIdx.x] = temp[blockDim.x * 2 - 1];
        temp[blockDim.x * 2 - 1] = 0;
    }
    __syncthreads();

    for (int d = blockDim.x; d >= 1; d /= 2) {
        int i = 2 * d * (tid + 1) - 1;
        if (i < blockDim.x * 2) {
            double val = temp[i - d];
            temp[i - d] = temp[i];
            temp[i] += val;
        }
        __syncthreads();
    }

    if (idx1 < n) d_out[idx1] = temp[tid];
    if (idx2 < n) d_out[idx2] = temp[tid + blockDim.x];
}

////////////////////////////////////////////////////////////////////////////////
// SCAN VERSION 2: Optimized Scan with Warp Shuffles
////////////////////////////////////////////////////////////////////////////////
__global__ void scan_shuffle_kernel(double* d_out, const double* d_in, double* d_block_sums, int n) {
    extern __shared__ double s_warp_sums[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    double val = (idx < n) ? d_in[idx] : 0.0;
    
    // Intra-warp exclusive scan
    double prefix_sum = 0;
    #pragma unroll
    for (int d = 1; d < WARP_SIZE; d *= 2) {
        double temp = __shfl_up_sync(0xFFFFFFFF, val, d);
        if (lane_id >= d) {
            prefix_sum += temp;
        }
    }
    
    // Last thread in warp holds total sum of the warp
    double warp_total = val + prefix_sum;
    if (lane_id == WARP_SIZE - 1) {
        s_warp_sums[warp_id] = warp_total;
    }
    __syncthreads();

    // First warp scans the warp sums
    if (warp_id == 0) {
        double warp_sum_val = (lane_id < (blockDim.x / WARP_SIZE)) ? s_warp_sums[lane_id] : 0.0;
        double warp_prefix = 0;
        #pragma unroll
        for (int d = 1; d < WARP_SIZE; d *= 2) {
            double temp = __shfl_up_sync(0xFFFFFFFF, warp_sum_val, d);
            if (lane_id >= d) {
                warp_prefix += temp;
            }
        }
        if (lane_id < (blockDim.x / WARP_SIZE)) {
            s_warp_sums[lane_id] = warp_prefix;
        }
    }
    __syncthreads();

    // Add scanned warp sums to each thread's local prefix sum
    double block_prefix = (warp_id > 0) ? s_warp_sums[warp_id] : 0.0;
    if (idx < n) {
        d_out[idx] = prefix_sum + block_prefix;
    }

    // Last thread writes total block sum
    if (threadIdx.x == blockDim.x - 1) {
        d_block_sums[blockIdx.x] = s_warp_sums[warp_id] + warp_total;
    }
}


////////////////////////////////////////////////////////////////////////////////
// CORRECTED RESAMPLING KERNEL
////////////////////////////////////////////////////////////////////////////////
__global__ void resample_kernel(Particle* d_new_particles, const Particle* d_old_particles, 
                                const double* d_cdf, curandState* states, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState localState = states[idx];
    double u = curand_uniform_double(&localState);

    // --- Perform a binary search (lower_bound) to find the particle ---
    // This is much more efficient than a linear scan and avoids kernel timeouts.
    int low = 0;
    int high = n - 1;
    int selected_idx = n - 1;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (d_cdf[mid] >= u) {
            selected_idx = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    d_new_particles[idx] = d_old_particles[selected_idx];
    states[idx] = localState;
}

////////////////////////////////////////////////////////////////////////////////
// Main host function
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <version>\n", argv[0]);
        fprintf(stderr, "  version: 1 for shared-memory scan, 2 for warp-shuffle scan\n");
        return 1;
    }
    int version = atoi(argv[1]);
    if (version != 1 && version != 2) {
        fprintf(stderr, "Error: version must be 1 or 2.\n");
        return 1;
    }
    printf("Initializing a 1D Particle Filter with %d particles.\n", NUM_PARTICLES);
    printf("Using Scan Version: %s\n", (version == 1) ? "Shared Memory (Blelloch)" : "Warp Shuffles");

    // --- Memory Allocation ---
    Particle *h_particles, *d_particles, *d_new_particles;
    double *d_weights, *d_cdf, *d_block_sums, *h_block_sums;
    curandState* d_rand_states;
    size_t particle_mem_size = sizeof(Particle) * NUM_PARTICLES;
    size_t double_mem_size = sizeof(double) * NUM_PARTICLES;
    size_t block_sum_mem_size = sizeof(double) * NUM_BLOCKS;
    h_particles = (Particle*)malloc(particle_mem_size);
    checkCudaErrors(cudaMalloc(&d_particles, particle_mem_size));
    checkCudaErrors(cudaMalloc(&d_new_particles, particle_mem_size));
    checkCudaErrors(cudaMalloc(&d_weights, double_mem_size));
    checkCudaErrors(cudaMalloc(&d_cdf, double_mem_size));
    checkCudaErrors(cudaMalloc(&d_block_sums, block_sum_mem_size));
    h_block_sums = (double*)malloc(block_sum_mem_size);
    checkCudaErrors(cudaMalloc(&d_rand_states, sizeof(curandState) * NUM_PARTICLES));

    // --- Initialization ---
    srand(123);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        h_particles[i].x = (rand() / (double)RAND_MAX - 0.5) * 2.0;
        h_particles[i].weight = 1.0 / NUM_PARTICLES;
    }
    checkCudaErrors(cudaMemcpy(d_particles, h_particles, particle_mem_size, cudaMemcpyHostToDevice));
    
    setup_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_rand_states, time(NULL), NUM_PARTICLES);
    getLastCudaError("cuRAND setup failed");

    // --- Main Filter Loop ---
    printf("Starting filter loop for 100 timesteps...\n");
    int num_timesteps = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int t = 0; t < num_timesteps; ++t) {
        predict_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_particles, d_rand_states, NUM_PARTICLES);
        double measurement = 5.0 * sin(t * 0.1);
        weight_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_particles, measurement, NUM_PARTICLES);
        
        extract_weights_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_particles, d_weights, NUM_PARTICLES);
        
        // --- SCAN ---
        if (version == 1) {
            size_t scan_shared_mem = sizeof(double) * NUM_THREADS * 2;
            scan_blelloch_kernel<<<NUM_BLOCKS, NUM_THREADS, scan_shared_mem>>>(d_cdf, d_weights, d_block_sums, NUM_PARTICLES);
        } else { // version == 2
            size_t scan_shared_mem = (NUM_THREADS / WARP_SIZE) * sizeof(double);
            scan_shuffle_kernel<<<NUM_BLOCKS, NUM_THREADS, scan_shared_mem>>>(d_cdf, d_weights, d_block_sums, NUM_PARTICLES);
        }
        
        checkCudaErrors(cudaMemcpy(h_block_sums, d_block_sums, block_sum_mem_size, cudaMemcpyDeviceToHost));
        double total_weight = 0;
        for(int i=0; i<NUM_BLOCKS; ++i) {
            double val = h_block_sums[i];
            h_block_sums[i] = total_weight;
            total_weight += val;
        }
        checkCudaErrors(cudaMemcpy(d_block_sums, h_block_sums, block_sum_mem_size, cudaMemcpyHostToDevice));
        
        add_block_sums_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_cdf, d_block_sums, NUM_PARTICLES);

        normalize_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_cdf, total_weight, NUM_PARTICLES);

        resample_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_new_particles, d_particles, d_cdf, d_rand_states, NUM_PARTICLES);

        Particle* temp = d_particles;
        d_particles = d_new_particles;
        d_new_particles = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Filter loop complete.\n");
    printf("Total GPU Time for %d iterations: %.3f ms\n", num_timesteps, milliseconds);
    printf("Time per iteration: %.5f ms\n", milliseconds / num_timesteps);

    // --- Cleanup ---
    free(h_particles);
    free(h_block_sums);
    checkCudaErrors(cudaFree(d_particles));
    checkCudaErrors(cudaFree(d_new_particles));
    checkCudaErrors(cudaFree(d_weights));
    checkCudaErrors(cudaFree(d_cdf));
    checkCudaErrors(cudaFree(d_block_sums));
    checkCudaErrors(cudaFree(d_rand_states));
    cudaDeviceReset();

    return 0;
}