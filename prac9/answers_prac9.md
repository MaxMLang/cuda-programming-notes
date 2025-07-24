### Practical 9 - Pattern Matching Notes

#### Core Concept - Integer-based String Matching
- The main trick is converting 4-character strings into single 32-bit `unsigned int`s.
- This turns slow string comparison into a single, fast integer comparison `(word == search_word)`.
- The main challenge is that words in the text don't always align perfectly with the 4-byte boundaries of the integers.
- The code has to handle this by checking the text at 4 different byte offsets (0, 1, 2, 3). This is done with bit-shifting to construct the unaligned words.

---

#### CPU vs GPU Implementation
- The CPU version (`gold_match`) is straightforward: loops through every integer of the text, then loops through the 4 offsets, then loops through the list of search words. Very sequential.
- The GPU version (`match_kernel`) parallelizes the search. The strategy is to make each thread responsible for finding just *one* of the search words.

---

#### GPU Kernel Details (`match_kernel.cu`)
- The biggest optimization is using shared memory to reduce slow global memory reads.
- The kernel processes the text in large `CHUNK_SIZE` blocks (we used 256 `int`s).
- All threads in the block work together to load one chunk of text from global memory into a shared memory array (`text_chunk`).
- To make the main loop cleaner, the threads also pre-calculate the 3 unaligned versions of the text chunk and store them in their own shared memory arrays (`offset1_chunk`, etc.).
- `__syncthreads()` is used multiple times to make sure all data is loaded and all offsets are calculated before the threads start searching. This prevents race conditions.
- After syncing, each thread loops through the shared memory chunks and compares the text against its single assigned word, incrementing a local counter.
- At the very end, each thread writes its final count to the results array in global memory.

---

#### Host Code (`main` function)
- The `main` function in `match.cu` handles all the setup and cleanup.
- Standard CUDA procedure:
    1. Allocate memory on the device (GPU) for the text, the search words, and the results array using `cudaMalloc`.
    2. Copy the input data (text, words) from the host (CPU) to the device (GPU) using `cudaMemcpy`.
    3. Launch the kernel: `match_kernel<<<1, nwords>>>(...)`. We used 1 block and 4 threads (since we have 4 search words).
    4. Copy the results array from the device back to the host.
    5. Free all GPU memory with `cudaFree`.

---

#### Results & Performance
- The output `CPU matches = 3 3 18 31` and `GPU matches = 3 3 18 31` confirms the GPU implementation is logically correct.
- This problem is well-suited for GPUs because the data access pattern is efficient. We read a chunk of memory once from the slow global memory and then process it many times over in fast shared memory.
- The performance gain would be much more obvious with a much larger text file and a much longer list of words to search for.