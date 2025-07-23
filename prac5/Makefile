# Define the directory for the executables
BIN_DIR := bin

# Define compiler and library paths
INC := -I$(CUDA_HOME)/include -I. -I../headers
LIB := -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcurand -lcufft

# Define NVCC compiler flags
NVCCFLAGS := -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

# Define the executables to be built
TARGETS := $(BIN_DIR)/tensorCUBLAS $(BIN_DIR)/simpleTensorCoreGEMM $(BIN_DIR)/compare_errors

# Default target to build all executables
all: $(TARGETS)

# Rule to create the bin directory
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Rule to build the tensorCUBLAS executable
$(BIN_DIR)/tensorCUBLAS: tensorCUBLAS.cu Makefile | $(BIN_DIR)
	nvcc $< -o $@ $(INC) $(NVCCFLAGS) $(LIB)

# Rule to build the simpleTensorCoreGEMM executable
$(BIN_DIR)/simpleTensorCoreGEMM: simpleTensorCoreGEMM.cu Makefile | $(BIN_DIR)
	nvcc $< -o $@ $(INC) $(NVCCFLAGS) $(LIB)

# Rule to build the compare_errors executable
$(BIN_DIR)/compare_errors: compare_errors.cu Makefile | $(BIN_DIR)
	nvcc $< -o $@ $(INC) $(NVCCFLAGS) $(LIB)

# Clean up the built files
clean:
	rm -rf $(BIN_DIR)