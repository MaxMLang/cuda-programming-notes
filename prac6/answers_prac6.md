### Practical 6 - Odds and Ends Notes

#### Separate Compilation (prac6 & prac6a)
* Main point was splitting up host code (main.cpp, compiled with g++) and device code (prac6.cu, compiled with nvcc).
* Used `extern "C"` in the C++ file to let it call the CUDA function. The linker connects them at the end.
* The Makefile showed two ways to do this:
    1.  Compiling both to object files (.o) and linking them.
    2.  Compiling the CUDA code into a static library (.a) and linking the main C++ code against that.
* Both methods produced identical executables (`prac6` and `prac6a`) and gave the same output, proving both linking methods work.

---

#### Templates for Data Types (prac6b)
* Used `template <class T>` to make a generic kernel that can work on different data types.
* This avoids writing the same kernel logic multiple times for `float`, `int`, `double`, etc.
* The compiler just "instantiates" a specific version when you call it, like `my_kernel<float><<<...>>>`.
* Modified the code to add a `double` precision version, which just involved adding the allocation, kernel call, and memory copy for the double arrays.

---

#### Templates for Values (prac6c)
* Used `template <int size>` to pass a constant integer value to the kernel at compile time.
* This is really useful for creating small, fixed-size arrays inside a kernel, like `float xl[size]`.
* The big advantage is that the compiler knows the exact size and can place these arrays in **registers**, which is the fastest memory on the GPU.
* Verified the output for the first thread (`tid=0`) was correct. It was 4 for the `size=2` kernel and 9 for the `size=3` kernel, as expected from the math.

---

#### Application Demo: Statistics on the GPU
* Wrote a new demo (`stats.cu`) to apply these concepts to a real-world problem.
* The goal was to calculate the mean and standard deviation of a large dataset.
* The kernel used a parallel **reduction** in shared memory to efficiently sum up all the values.
* The final C++ `main` file generated the data, called the external CUDA `calculate_stats` function, and then verified the GPU's result against a CPU calculation.
* It was a good example of putting it all together: a C++ host program delegating a heavy parallel task to a separate, specialized CUDA function.