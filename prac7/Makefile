# Makefile for Practical 7

# Compiler
NVCC := nvcc

# Directories
CUDA_HOME ?= /usr/local/cuda
BIN_DIR := bin
SRC_DIR := .

# Includes and Libraries
INC := -I$(CUDA_HOME)/include -I. -I$(SRC_DIR)/../headers
LIB := -L$(CUDA_HOME)/lib64 -lcudart

# Compiler flags
NVCCFLAGS := -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

# Source files
TRID_DYN_SRC := $(SRC_DIR)/trid_dyn.cu $(SRC_DIR)/trid_gold.cpp
TRID_MULTI_SRC := $(SRC_DIR)/trid_multi.cu $(SRC_DIR)/trid_multi_gold.cpp

# Executables
TRID_DYN_EXE := $(BIN_DIR)/trid_dyn
TRID_MULTI_EXE := $(BIN_DIR)/trid_multi

# Targets
all: $(TRID_DYN_EXE) $(TRID_MULTI_EXE)

# Rule to create the bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule to build the dynamic shared memory version
$(TRID_DYN_EXE): $(TRID_DYN_SRC) | $(BIN_DIR)
	$(NVCC) $(TRID_DYN_SRC) -o $@ $(INC) $(NVCCFLAGS) $(LIB)

# Rule to build the multi-system version
$(TRID_MULTI_EXE): $(TRID_MULTI_SRC) | $(BIN_DIR)
	$(NVCC) $(TRID_MULTI_SRC) -o $@ $(INC) $(NVCCFLAGS) $(LIB)

# Clean rule
clean:
	rm -rf $(BIN_DIR)