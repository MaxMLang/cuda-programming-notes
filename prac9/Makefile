# Define CUDA home and libraries
CUDA_HOME   ?= /usr/local/cuda
INC         := -I$(CUDA_HOME)/include -I.
LIB         := -L$(CUDA_HOME)/lib64 -lcudart

# Compiler flags
NVCCFLAGS   := -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

# Target executable
TARGET      := match
BIN_DIR     := bin
EXECUTABLE  := $(BIN_DIR)/$(TARGET)

# Source files
SOURCES     := match.cu match_gold.cpp

# Default target
all: $(EXECUTABLE)

# Rule to build the executable
$(EXECUTABLE): $(SOURCES) | $(BIN_DIR)
	nvcc $(SOURCES) -o $@ $(INC) $(NVCCFLAGS) $(LIB)

# Rule to create the bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Clean up build files
clean:
	rm -rf $(BIN_DIR) $(TARGET)
