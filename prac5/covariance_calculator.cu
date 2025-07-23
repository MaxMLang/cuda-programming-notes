#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// Function to check for CUDA errors
#define checkCuda(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Function to check for cuBLAS errors
#define checkCublas(status) { \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUBLAS Error at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA Kernel to subtract the mean from each column (feature)
__global__ void subtract_mean_kernel(float* data, const float* means, int n, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * d) {
        int col = idx % d;
        data[idx] -= means[col];
    }
}

int main() {
    // --- 1. Setup ---
    int n = 4096; // Number of samples
    int d = 1024; // Number of features (variables)

    printf("Calculating %dx%d covariance matrix for %d samples with %d features.\n", d, d, n, d);

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));
    
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // --- 2. Host Memory Allocation and Data Initialization ---
    std::vector<float> h_data(n * d);
    for (int i = 0; i < n * d; ++i) {
        h_data[i] = (float)(rand() % 100);
    }

    std::vector<float> h_means(d, 0.0f);
    // Calculate means on the CPU for simplicity
    for (int j = 0; j < d; ++j) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += h_data[i * d + j];
        }
        h_means[j] = (float)(sum / n);
    }

    // --- 3. Device Memory Allocation and Data Transfer ---
    float *d_data, *d_means, *d_cov;
    checkCuda(cudaMalloc((void**)&d_data, n * d * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_means, d * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_cov, d * d * sizeof(float)));

    checkCuda(cudaMemcpy(d_data, h_data.data(), n * d * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_means, h_means.data(), d * sizeof(float), cudaMemcpyHostToDevice));

    // --- 4. GPU Computation ---
    checkCuda(cudaEventRecord(start));

    // Step A: Center the data (subtract mean from each feature column)
    dim3 blockDim(256);
    dim3 gridDim((n * d + blockDim.x - 1) / blockDim.x);
    subtract_mean_kernel<<<gridDim, blockDim>>>(d_data, d_means, n, d);
    checkCuda(cudaGetLastError()); // Check for kernel launch errors

    // Step B: Calculate Covariance Matrix: C = (1/(n-1)) * A^T * A
    // Here, A is the mean-centered data matrix.
    const float alpha = 1.0f / (n - 1);
    const float beta = 0.0f;
    
    // We compute C = alpha * (d_data^T * d_data) + beta * C
    // The dimensions for the GEMM operation are: (d x n) * (n x d) -> (d x d)
    // CUBLAS expects column-major, so we treat our (n x d) matrix as such.
    // The GEMM arguments become: m=d, n=d, k=n
    checkCublas(cublasSgemm(handle,
                            CUBLAS_OP_T,     // Transpose A
                            CUBLAS_OP_N,     // No Transpose B
                            d, d, n,         // m, n, k
                            &alpha,          // alpha
                            d_data, n,       // A and lda (leading dimension)
                            d_data, n,       // B and ldb
                            &beta,           // beta
                            d_cov, d));      // C and ldc

    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("GPU Covariance Matrix calculation complete.\n");
    printf("Time taken: %f ms\n\n", milliseconds);

    // --- 5. Cleanup ---
    cudaFree(d_data);
    cudaFree(d_means);
    cudaFree(d_cov);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}