# Makefile for Laplace 3D Practical

# Compiler and flags
NVCC        := nvcc
NVCCFLAGS   := -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

# Directories and paths
# Assumes CUDA_HOME is an environment variable. If not, set it here.
# e.g., CUDA_HOME ?= /apps/system/easybuild/software/CUDA/12.6.0
INCLUDES    := -I$(CUDA_HOME)/include -I. -I../headers
LIB         := -L$(CUDA_HOME)/lib64 -lcudart

# Build directory for all output files
BUILD_DIR   := bin

# List of executables for the 'all' target
EXECUTABLES := $(BUILD_DIR)/laplace3d \
               $(BUILD_DIR)/laplace3d_new \
               $(BUILD_DIR)/laplace3d_benchmark

# Phony targets do not represent files.
.PHONY: all clean benchmark

# The default goal when you just type 'make'
all: $(EXECUTABLES)

# --- Build Rules ---

# Rule for executables with two source files
$(BUILD_DIR)/laplace3d: laplace3d.cu laplace3d_gold.cpp
	@mkdir -p $(@D)
	@echo "Building $@..."
	$(NVCC) $^ -o $@ $(INCLUDES) $(NVCCFLAGS) $(LIB)

$(BUILD_DIR)/laplace3d_new: laplace3d_new.cu laplace3d_gold.cpp
	@mkdir -p $(@D)
	@echo "Building $@..."
	$(NVCC) $^ -o $@ $(INCLUDES) $(NVCCFLAGS) $(LIB)

# Rule for executables with one source file
$(BUILD_DIR)/laplace3d_benchmark: laplace3d_benchmark.cu
	@mkdir -p $(@D)
	@echo "Building $@..."
	$(NVCC) $< -o $@ $(INCLUDES) $(NVCCFLAGS) $(LIB)


# --- Benchmark Rule ---

# This target depends on the benchmark executable being built first.
# It then runs all the required tests sequentially.
benchmark: $(BUILD_DIR)/laplace3d_benchmark
	@echo ""
	@echo "--- Running Benchmarks (1024x1024x1024 grid) ---"
	@echo "--- Benchmarking v1 Kernel (Original 2D block) ---"
	$(BUILD_DIR)/laplace3d_benchmark 1 16 16 0
	$(BUILD_DIR)/laplace3d_benchmark 1 32 8 0
	$(BUILD_DIR)/laplace3d_benchmark 1 8 32 0
	$(BUILD_DIR)/laplace3d_benchmark 1 32 16 0
	@echo ""
	@echo "--- Benchmarking v2 Kernel (New 3D block) ---"
	$(BUILD_DIR)/laplace3d_benchmark 2 8 8 8
	$(BUILD_DIR)/laplace3d_benchmark 2 4 8 16
	$(BUILD_DIR)/laplace3d_benchmark 2 4 16 16
	$(BUILD_DIR)/laplace3d_benchmark 2 8 8 16
	@echo "--- Benchmarks Complete ---"
	@echo ""


# --- Cleanup Rule ---

clean:
	@echo "Cleaning up..."
	rm -rf $(BUILD_DIR)

