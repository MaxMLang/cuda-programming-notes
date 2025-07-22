# Makefile for Practical 2: Monte Carlo

# Compiler and flags
NVCC        := nvcc
NVCCFLAGS   := -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

# Directories and paths
# Assumes CUDA_HOME is an environment variable. If not, set it here.
# e.g., CUDA_HOME ?= /apps/system/easybuild/software/CUDA/12.6.0
INCLUDES    := -I$(CUDA_HOME)/include -I. -I../headers

# Libraries - Added -lcurand for the Random Number Generator library
LIB         := -L$(CUDA_HOME)/lib64 -lcudart -lcurand

# Build directory for all output files
BUILD_DIR   := bin

# List of all executables to build
EXECUTABLES := $(BUILD_DIR)/prac2 \
               $(BUILD_DIR)/prac2_device \
               $(BUILD_DIR)/prac2_benchmark

# Phony targets do not represent files
.PHONY: all clean

# The default goal when you just type 'make'
all: $(EXECUTABLES)

# --- Build Rules ---

# This is a generic pattern rule. It says how to build any file in the
# BUILD_DIR from a corresponding .cu file in the current directory.
# $@ is the target file (e.g., bin/prac2)
# $< is the first prerequisite (e.g., prac2.cu)
# $(@D) is the directory part of the target (e.g., bin)
$(BUILD_DIR)/%: %.cu
	@mkdir -p $(@D)
	@echo "Building $@..."
	$(NVCC) $< -o $@ $(INCLUDES) $(NVCCFLAGS) $(LIB)

# --- Cleanup Rule ---

clean:
	@echo "Cleaning up..."
	rm -rf $(BUILD_DIR)

