# Get the CUDA home directory
CUDA_HOME := $(shell dirname $(shell dirname `which nvcc`))

# Compiler flags
INC     := -I$(CUDA_HOME)/include -I.
LIB     := -L$(CUDA_HOME)/lib64 -lcudart
CXXFLAGS := -fPIC -std=c++11 -O2
NVCC_FLAGS := -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math $(INC)

# List of all executables
TARGETS := prac6 prac6a prac6b prac6c stats

# Default target
all: $(addprefix bin/, $(TARGETS))

# --- Rules for original practicals ---
bin/prac6: obj/main.o obj/prac6.o
	g++ $(CXXFLAGS) -o $@ $^ $(LIB)

bin/prac6a: obj/main.o obj/prac6.a
	g++ $(CXXFLAGS) -o $@ $^ $(LIB)

bin/prac6b: main.cpp prac6b.cu
	nvcc $(NVCC_FLAGS) -o $@ $^ $(LIB)

bin/prac6c: main.cpp prac6c.cu
	nvcc $(NVCC_FLAGS) -o $@ $^ $(LIB)

# Object files for original practicals
obj/main.o: main.cpp
	g++ $(CXXFLAGS) -c -o $@ $<

obj/prac6.o: prac6.cu
	nvcc $(NVCC_FLAGS) -c -o $@ $<

obj/prac6.a: prac6.cu
	nvcc $(NVCC_FLAGS) -lib -o $@ $<


# --- Rules for the new stats program ---
bin/stats: obj/main_stats.o obj/stats.o
	g++ $(CXXFLAGS) -o $@ $^ $(LIB)

obj/main_stats.o: main_stats.cpp
	g++ $(CXXFLAGS) -c -o $@ $<

obj/stats.o: stats.cu
	nvcc $(NVCC_FLAGS) -c -o $@ $<


# --- Housekeeping rules ---
.PHONY: all clean

# Create directories
$(shell mkdir -p bin obj)

clean:
	rm -rf bin obj *~