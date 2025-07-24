#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
# set custom output and error file names
#SBATCH --output=%x_%j.out
# set name of job
#SBATCH --job-name=prac7

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
echo "--- Running trid_dyn (Dynamic Shared Memory) ---"
./bin/trid_dyn
echo ""

echo "--- Running trid_multi (Multiple Systems) ---"
./bin/trid_multi
echo ""