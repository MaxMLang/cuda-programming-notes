#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#
# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#
# set custom output and error file names
#SBATCH --output=%x_%j.out
#
# set name of job
#SBATCH --job-name=prac9
#
# use our reservation if available
#SBATCH --reservation=cuda2025

# Purge existing modules and load CUDA
module purge
module load CUDA

# Clean previous builds and compile the program
echo "--- Cleaning and building project... ---"
make clean
make
echo "--- Build complete. ---"
echo ""

# Execute the program and label its output
echo "--- Running pattern matching program ---"
./bin/match
echo ""

echo "--- Job complete ---"
