// main_stats.cpp
// Host code to run a statistics calculation on the GPU.

#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <cmath>
#include <iomanip>

// Promise to the compiler that this function exists in another file.
extern "C" void calculate_stats_gpu(const float* data, size_t size, float* mean, float* std_dev);

int main() {
    const size_t N = 1 << 20; // A large dataset (2^20 = 1,048,576 elements)

    // --- 1. Generate Sample Data on the Host (CPU) ---
    std::vector<float> h_data(N);
    std::mt19937 rng(123); // Mersenne Twister random number generator with a fixed seed
    std::normal_distribution<float> dist(100.0f, 15.0f); // Mean=100, StdDev=15

    for (size_t i = 0; i < N; ++i) {
        h_data[i] = dist(rng);
    }
    std::cout << "Generated " << N << " random numbers." << std::endl;

    // --- 2. Run Calculation on the GPU ---
    float gpu_mean = 0.0f;
    float gpu_std_dev = 0.0f;
    calculate_stats_gpu(h_data.data(), N, &gpu_mean, &gpu_std_dev);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[GPU Result] Mean: " << gpu_mean << ", Standard Deviation: " << gpu_std_dev << std::endl;


    // --- 3. Verify Result on the CPU ---
    double cpu_sum = 0.0;
    for(size_t i = 0; i < N; ++i) {
        cpu_sum += h_data[i];
    }
    float cpu_mean = static_cast<float>(cpu_sum / N);

    double cpu_sq_diff_sum = 0.0;
    for(size_t i = 0; i < N; ++i) {
        cpu_sq_diff_sum += (h_data[i] - cpu_mean) * (h_data[i] - cpu_mean);
    }
    float cpu_std_dev = static_cast<float>(std::sqrt(cpu_sq_diff_sum / N));

    std::cout << "[CPU Result] Mean: " << cpu_mean << ", Standard Deviation: " << cpu_std_dev << std::endl;

    // --- 4. Check for differences ---
    if (std::abs(gpu_mean - cpu_mean) < 1e-4 && std::abs(gpu_std_dev - cpu_std_dev) < 1e-4) {
        std::cout << "\n✅ Verification PASSED!" << std::endl;
    } else {
        std::cout << "\n❌ Verification FAILED!" << std::endl;
    }

    return 0;
}