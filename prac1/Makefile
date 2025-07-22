INC := -I$(CUDA_HOME)/include -I. -I../headers
LIB := -L$(CUDA_HOME)/lib64 -lcudart
NVCCFLAGS := -lineinfo -arch=sm_70 --ptxas-options=-v --use_fast_math

all: prac1a prac1b prac1c

prac1a: prac1a.cu Makefile
	mkdir -p ./bin
	nvcc prac1a.cu -o ./bin/prac1a $(INC) $(NVCCFLAGS) $(LIB)

prac1b: prac1b.cu Makefile
	mkdir -p ./bin
	nvcc prac1b.cu -o ./bin/prac1b $(INC) $(NVCCFLAGS) $(LIB)

prac1c: prac1c.cu Makefile
	mkdir -p ./bin
	nvcc prac1c.cu -o ./bin/prac1c $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -rf ./bin
