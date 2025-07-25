#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
# set custom output and error file names
#SBATCH --output=%x_%j.out   # Creates a file like "prac5_5388446.out"
#SBATCH --error=%x_%j.err    # Creates a file like "prac5_5388446.err"
# set name of job
#SBATCH --job-name=prac5

# use our reservation
#SBATCH --reservation=cuda2025

# Purge existing modules and load CUDA
module purge
module load CUDA

# Clean previous builds and compile all programs
echo "--- Cleaning and building projects... ---"
make clean
make
echo "--- Build complete. ---"
echo ""

# Execute the programs and label their output
echo "--- Running simpleTensorCoreGEMM ---"
./bin/simpleTensorCoreGEMM
echo ""

echo "--- Running tensorCUBLAS ---"
./bin/tensorCUBLAS
echo ""

echo "--- Running compare_errors ---"
./bin/compare_errors
echo ""

echo "--- Running covariance_calculator ---"
./bin/covariance_calculator
echo ""