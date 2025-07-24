#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

# set custom output and error file names
# The %x will be replaced by the job name, %j by the job ID.
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# set name of job
#SBATCH --job-name=prac8

# use our reservation if available
#SBATCH --reservation=cuda2025

# Purge existing modules and load the required CUDA module
module purge
module load CUDA

# Create a fresh build
echo "--- Cleaning and building projects... ---"
make clean
make
echo "--- Build complete. ---"
echo ""

# Check if the bin directory and executables exist
if [ ! -d "bin" ]; then
    echo "Error: bin directory not found. Build may have failed."
    exit 1
fi

# Execute the programs and label their output
echo "--- Running scan_multiblock ---"
if [ -f "bin/scan_multiblock" ]; then
    ./bin/scan_multiblock
else
    echo "Error: scan_multiblock not found."
fi
echo ""

echo "--- Running scan_shuffle ---"
if [ -f "bin/scan_shuffle" ]; then
    ./bin/scan_shuffle
else
    echo "Error: scan_shuffle not found."
fi
echo ""

echo "--- Running scan_recurrence ---"
if [ -f "bin/scan_recurrence" ]; then
    ./bin/scan_recurrence
else
    echo "Error: scan_recurrence not found."
fi
echo ""

echo "--- All jobs finished. ---"
