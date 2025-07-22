# Define CUDA home and common flags
CUDA_HOME   := $(shell dirname $(shell dirname `which nvcc`))
INC         := -I$(CUDA_HOME)/include -I. -I../headers
LIB         := -L$(CUDA_HOME)/lib64 -lcudart -lcurand
NVCCFLAGS   := -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

# --- Targets ---

# Default target builds all executables
all: reduction laplace3d_rms particle_filter_benchmark

# Builds the reduction executable
reduction: reduction.cu
	@mkdir -p ./bin
	nvcc reduction.cu -o ./bin/reduction $(INC) $(NVCCFLAGS) $(LIB)

# Builds the laplace3d executable with RMS calculation
laplace3d_rms: laplace3d_rms.cu
	@mkdir -p ./bin
	nvcc laplace3d_rms.cu -o ./bin/laplace3d_rms $(INC) $(NVCCFLAGS) $(LIB)

# Builds the particle filter benchmark executable
particle_filter_benchmark: particle_filter_benchmark.cu
	@mkdir -p ./bin
	nvcc particle_filter_benchmark.cu -o ./bin/particle_filter_benchmark $(INC) $(NVCCFLAGS) $(LIB)

# Removes compiled binaries
clean:
	rm -f ./bin/reduction ./bin/laplace3d_rms ./bin/particle_filter_benchmark
