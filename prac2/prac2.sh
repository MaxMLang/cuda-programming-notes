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

./bin/prac2
./bin/prac2_device
./bin/prac2_benchmark 1 128
./bin/prac2_benchmark 2 128