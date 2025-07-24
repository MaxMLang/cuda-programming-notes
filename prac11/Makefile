# Makefile for Practical 11

# Compiler
NVCC = nvcc

# Compiler flags
# -g for debugging, -O3 for optimization
NVCC_FLAGS = -O3 -g -arch=sm_70

# Per-thread default stream flag
PER_THREAD_FLAG = --default-stream per-thread

# Source files
SOURCES = stream_test.cu multithread_test.cu stream_test_prints.cu overlapped_processing.cu statistics_example.cu

# Executable names
EXECS = stream_legacy stream_per_thread \
        multithread_legacy multithread_per_thread \
        stream_prints_legacy stream_prints_per_thread \
        overlapped_processing \
        statistics_example

# Target directory
BIN_DIR = bin

# Default target
all: $(addprefix $(BIN_DIR)/, $(EXECS))

# Rule to create the bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# --- Stream Test Rules ---
$(BIN_DIR)/stream_legacy: stream_test.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BIN_DIR)/stream_per_thread: stream_test.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(PER_THREAD_FLAG) -o $@ $<

# --- Multithread Test Rules ---
$(BIN_DIR)/multithread_legacy: multithread_test.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -Xcompiler -fopenmp -o $@ $<

$(BIN_DIR)/multithread_per_thread: multithread_test.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(PER_THREAD_FLAG) -Xcompiler -fopenmp -o $@ $<
	
# --- Stream Test with Prints Rules ---
$(BIN_DIR)/stream_prints_legacy: stream_test_prints.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

$(BIN_DIR)/stream_prints_per_thread: stream_test_prints.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(PER_THREAD_FLAG) -o $@ $<

# --- Overlapped Processing Rule (needs per-thread flag to be effective) ---
$(BIN_DIR)/overlapped_processing: overlapped_processing.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(PER_THREAD_FLAG) -o $@ $<

# --- Statistics Example Rule (benefits from per-thread flag) ---
$(BIN_DIR)/statistics_example: statistics_example.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(PER_THREAD_FLAG) -o $@ $< -lcurand

# Clean rule
clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
