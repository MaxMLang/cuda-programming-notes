#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=prac4

# use our reservation
#SBATCH --reservation=cuda2025

module purge
module load CUDA

make clean
make

./bin/reduction
./bin/laplace3d_rms
echo "Running particle filter benchmark...  1. A shared-memory-based Blelloch scan."
./bin/particle_filter_benchmark 1
echo "2. An optimized scan using warp shuffle instructions..."
./bin/particle_filter_benchmark 2