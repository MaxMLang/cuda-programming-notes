//
// CUDA Kernel for Pattern Matching
//

// Define the size of the text chunk to process in shared memory
#define CHUNK_SIZE 256

__global__ void match_kernel(unsigned int *text, unsigned int *words, int *matches, int nwords, int len)
{
    // Shared memory for the text chunk and its offset versions
    __shared__ unsigned int text_chunk[CHUNK_SIZE];
    __shared__ unsigned int offset1_chunk[CHUNK_SIZE];
    __shared__ unsigned int offset2_chunk[CHUNK_SIZE];
    __shared__ unsigned int offset3_chunk[CHUNK_SIZE];

    // Each thread is responsible for one word
    int w = threadIdx.x;

    // Local counter for matches found by this thread
    int local_matches = 0;

    // The specific word this thread is searching for
    unsigned int my_word = words[w];

    // Process the entire text in chunks
    for (int i = 0; i < len; i += CHUNK_SIZE) {

        // Cooperate to load the text chunk into shared memory
        // Each thread loads a part of the chunk
        for (int j = w; j < CHUNK_SIZE; j += blockDim.x) {
            int global_idx = i + j;
            if (global_idx < len) {
                text_chunk[j] = text[global_idx];
            }
        }
        __syncthreads(); // Wait for all threads to finish loading

        // Cooperate to compute and store the 3 offset versions
        for (int j = w; j < CHUNK_SIZE; j += blockDim.x) {
            int global_idx = i + j;
            if (global_idx < len -1) {
                offset1_chunk[j] = (text[global_idx] >> 8)  | (text[global_idx+1] << 24);
                offset2_chunk[j] = (text[global_idx] >> 16) | (text[global_idx+1] << 16);
                offset3_chunk[j] = (text[global_idx] >> 24) | (text[global_idx+1] << 8);
            }
        }
        __syncthreads(); // Wait for all threads to finish computing offsets

        // Now, each thread searches for its word in the shared memory chunk
        if (w < nwords) {
            for (int j = 0; j < CHUNK_SIZE; j++) {
                if (i + j < len) {
                    local_matches += (text_chunk[j]    == my_word);
                    local_matches += (offset1_chunk[j] == my_word);
                    local_matches += (offset2_chunk[j] == my_word);
                    local_matches += (offset3_chunk[j] == my_word);
                }
            }
        }
        __syncthreads(); // Ensure all threads are done before loading the next chunk
    }

    // Each thread writes its final count to the global results array
    if (w < nwords) {
        matches[w] = local_matches;
    }
}
