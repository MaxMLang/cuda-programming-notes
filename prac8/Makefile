# CUDA toolkit installation path
CUDA_HOME   ?= /usr/local/cuda

# Common includes and libraries
INC := -I$(CUDA_HOME)/include -I.
LIB := -L$(CUDA_HOME)/lib64 -lcudart

# NVCC compiler flags
# sm_70 is for Volta, sm_75 for Turing, sm_86 for Ampere. Adjust to your GPU.
# Use --use_fast_math for performance, but be aware of potential precision loss.
NVCCFLAGS   := -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

# Location for helper files from the CUDA samples (if you use them)
# For this practical, we assume helper_cuda.h is in the same directory.
# If not, adjust this path.
HELPER_PATH ?= .
INC += -I$(HELPER_PATH)

# List of executables to build
TARGETS := scan_multiblock scan_shuffle scan_recurrence

# Output directory for binaries
BIN_DIR := bin

# The default goal is to build all targets
all: $(patsubst %,$(BIN_DIR)/%,$(TARGETS))

# Rule to create the bin directory
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Generic rule to compile a .cu file into an executable in the bin directory
$(BIN_DIR)/%: %.cu Makefile | $(BIN_DIR)
	@echo "--- Compiling $< -> $@ ---"
	$(CUDA_HOME)/bin/nvcc $< -o $@ $(INC) $(NVCCFLAGS) $(LIB)

# Clean up build files
clean:
	@echo "--- Cleaning project files ---"
	rm -rf $(BIN_DIR)
	rm -f *.o

.PHONY: all clean
