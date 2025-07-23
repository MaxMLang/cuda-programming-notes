### Practical 5 - Tensor Cores & cuBLAS Notes

#### `tensorCUBLAS` - Easy Tensor Core Access
* This practical was about using Tensor Cores, special GPU hardware for fast matrix math.
* The first code, `tensorCUBLAS`, shows how to use the cuBLAS library to get a big speedup easily.
* The code runs a matrix multiply three ways: standard FP32, TF32 for newer GPUs, and mixed-precision with FP16 inputs.
* I ran tests for matrix sizes from n=32 up to n=1024.
* Results clearly show using Tensor Cores is way faster. The first timing for each run is always a "warm-up" and much slower, so I ignored it. For n=1024, the speedup was huge:
    * No Tensor Cores: ~0.27 ms
    * Mixed-Precision (FP16): ~0.08 ms
* The performance scaling seems to be about what you'd expect for matrix multiplication, which is roughly $O(n^3)$. The graph of time vs n would curve up sharply, but the curve for the Tensor Core version would be much flatter.

---

#### `simpleTensorCoreGEMM` - Manual WMMA
* The second code demonstrates how to program Tensor Cores directly inside a CUDA kernel, which is a lot harder.
* It compares its own result against the cuBLAS library to check for errors.
* Initially, it can fail because half-precision (FP16) math isn't very accurate]. I had to increase the allowed error margin (`eps`) from `1e-4` to `1e-3` to get it to pass.
* The results for the massive 16384x16384 matrix show that even a simple manual implementation is much slower than the official library:
    * WMMA Kernel Time: ~893 ms
    * cuBLAS Time: ~142 ms
* This really proves that for performance, you should just use cuBLAS unless you have a very specific reason not to.
* The kernel works by having each warp compute a small $16 \times 16$ tile of the final matrix. The whole thread block works together on a bigger $64 \times 64$ tile

---

#### `compare_errors` - Precision vs. Performance
* I wrote a new script to properly check the numerical errors between the different methods.
* It calculates a "ground truth" result on the CPU with `double` precision.
* It then compares the three GPU methods (No TC, TF32, FP16) against this CPU result.
* Key finding: As you increase performance by using lower precision, the error gets bigger.
    * FP32 (no TCs) is very accurate.
    * TF32 is a bit less accurate but faster.
    * FP16 mixed-precision is the fastest but has the largest (though still small) error.
* This shows the main trade-off: you sacrifice some numerical precision for a big gain in speed.

#### covariance_calculator - A Statistical Application

* Added a final code to show how this stuff is useful in statistics.
* It calculates a covariance matrix, which is a super common thing to do with data.
* The core of the calculation is just mean-centering the data and then doing a big matrix multiplication (AT×A), which is perfect for cublasSgemm.
