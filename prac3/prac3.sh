#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=prac2

# use our reservation
#SBATCH --reservation=cuda2025

module purge
module load CUDA

make clean
make

./bin/laplace3d
./bin/laplace3d_new
echo "Running benchmark... v1 Kernel"
./bin/laplace3d_benchmark 1 16 16 0
./bin/laplace3d_benchmark 1 32 8 0
./bin/laplace3d_benchmark 1 8 32 0
./bin/laplace3d_benchmark 1 32 16 0

echo "Running benchmark... v2 Kernel"
./bin/laplace3d_benchmark 2 8 8 8
./bin/laplace3d_benchmark 2 4 8 16
./bin/laplace3d_benchmark 2 4 16 16
./bin/laplace3d_benchmark 2 8 8 16
