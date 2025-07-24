# Makefile for CUDA Practical 12

# Compiler
NVCC = nvcc

# Compiler flags
# -O3 is for optimization
# -arch=sm_75 and -gencode=arch=compute_75,code=sm_75 are for targeting Turing architecture.
NVCCFLAGS = -O3 -arch=sm_70 -gencode=arch=compute_70,code=sm_70

# Path to the CUDA helper files (if they are in a different directory)
# For this example, we assume they are in the same directory or in the include path.
LDFLAGS =

# Directory for the output executables
BIN_DIR = bin

# List of source files
SRCS = kernel_overlap.cu work_streaming.cu work_streaming_solution.cu bootstrap_ci.cu

# Generate the list of target executables in the bin directory
TARGETS = $(patsubst %.cu, $(BIN_DIR)/%, $(SRCS))

# The default 'all' target builds everything
all: $(TARGETS)

# Rule to create the bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule to build an executable from a .cu file
# $< is the first prerequisite (the .cu file)
# $@ is the target (the executable in bin/)
# The `| $(BIN_DIR)` part ensures the bin directory is created before compiling.
$(BIN_DIR)/%: %.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

# The 'clean' target removes the bin directory and all its contents
clean:
	rm -rf $(BIN_DIR)

# Phony targets are not files
.PHONY: all clean
