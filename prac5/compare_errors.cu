#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm> // For std::max

// A small value to prevent division by zero in relative error calculation
#define EPSILON 1e-12

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

/**
 * @brief Analyzes and prints error metrics between a reference and a test matrix.
 * @param ref The ground truth matrix (e.g., CPU result).
 * @param test The matrix to test (e.g., GPU result).
 * @param name The name of the method being tested.
 * @param n The dimension of the square matrix.
 */
void analyze_errors(const std::vector<double>& ref, const std::vector<float>& test, const std::string& name, int n) {
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    
    for (int i = 0; i < n * n; ++i) {
        double abs_err = fabs((double)test[i] - ref[i]);
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
        }

        if (fabs(ref[i]) > EPSILON) {
            double rel_err = abs_err / fabs(ref[i]);
            if (rel_err > max_rel_err) {
                max_rel_err = rel_err;
            }
        }
    }
    
    printf("--- Analysis for: %s ---\n", name.c_str());
    printf("  Max Absolute Error: %e\n", max_abs_err);
    printf("  Max Relative Error: %e\n", max_rel_err);
    printf("---------------------------------------\n\n");
}


int main(void) {
    // --- Setup ---
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    int n = 1024; // Matrix size
    size_t matrix_bytes = n * n * sizeof(float);
    
    float alpha = 1.0f;
    float beta = 0.0f;

    // --- Host Memory ---
    std::vector<float> h_A(n * n);
    std::vector<float> h_B(n * n);
    std::vector<double> h_C_ref(n * n, 0.0); // CPU ground truth in double precision
    std::vector<float> h_C_test(n * n);      // Buffer for GPU results

    // Use FP16 for the mixed-precision test
    std::vector<__half> h_A_h(n * n);
    std::vector<__half> h_B_h(n * n);

    // Initialise matrices with random data
    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
        h_A_h[i] = h_A[i];
        h_B_h[i] = h_B[i];
    }
    
    // --- Device Memory ---
    float *d_A, *d_B, *d_C;
    __half *d_A_h, *d_B_h;
    checkCuda(cudaMalloc((void**)&d_A, matrix_bytes));
    checkCuda(cudaMalloc((void**)&d_B, matrix_bytes));
    checkCuda(cudaMalloc((void**)&d_C, matrix_bytes));
    checkCuda(cudaMalloc((void**)&d_A_h, n * n * sizeof(__half)));
    checkCuda(cudaMalloc((void**)&d_B_h, n * n * sizeof(__half)));
    
    // Copy input data to device
    checkCuda(cudaMemcpy(d_A, h_A.data(), matrix_bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B.data(), matrix_bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_A_h, h_A_h.data(), n * n * sizeof(__half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B_h, h_B_h.data(), n * n * sizeof(__half), cudaMemcpyHostToDevice));
    
    // 1. --- Calculate Ground Truth on CPU ---
    printf("Calculating ground truth on CPU (this may take a moment)...\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += (double)h_A[i * n + k] * (double)h_B[k * n + j];
            }
            h_C_ref[i * n + j] = sum;
        }
    }
    printf("CPU calculation complete.\n\n");

    // 2. --- Test 1: SGEMM without Tensor Cores ---
    checkCublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B, n, d_A, n, &beta, d_C, n));
    checkCuda(cudaMemcpy(h_C_test.data(), d_C, matrix_bytes, cudaMemcpyDeviceToHost));
    analyze_errors(h_C_ref, h_C_test, "SGEMM (No Tensor Cores, FP32)", n);

    // 3. --- Test 2: SGEMM with TF32 Tensor Cores ---
    checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B, n, d_A, n, &beta, d_C, n));
    checkCuda(cudaMemcpy(h_C_test.data(), d_C, matrix_bytes, cudaMemcpyDeviceToHost));
    analyze_errors(h_C_ref, h_C_test, "SGEMM (TF32 Tensor Cores)", n);

    // 4. --- Test 3: SGEMM with Mixed-Precision Tensor Cores ---
    checkCublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    checkCublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B_h, CUDA_R_16F, n, d_A_h, CUDA_R_16F, n, &beta, d_C, CUDA_R_32F, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    checkCuda(cudaMemcpy(h_C_test.data(), d_C, matrix_bytes, cudaMemcpyDeviceToHost));
    analyze_errors(h_C_ref, h_C_test, "SGEMM (Mixed-Precision FP16)", n);

    // --- Cleanup ---
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A_h); cudaFree(d_B_h);
    cublasDestroy(handle);

    return 0;
}