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
#SBATCH --job-name=prac11

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

# --- Execute the programs and label their output ---

echo "--- Running stream_legacy ---"
./bin/stream_legacy
echo ""

echo "--- Running stream_per_thread (expecting much faster time) ---"
./bin/stream_per_thread
echo ""

echo "--- Running multithread_legacy ---"
./bin/multithread_legacy
echo ""

echo "--- Running multithread_per_thread (expecting much faster time) ---"
./bin/multithread_per_thread
echo ""

echo "--- Running stream_prints_legacy (output will be interleaved) ---"
./bin/stream_prints_legacy
echo ""

echo "--- Running stream_prints_per_thread (kernels should start/finish concurrently) ---"
./bin/stream_prints_per_thread
echo ""

echo "--- Running overlapped_processing ---"
./bin/overlapped_processing
echo ""

echo "--- Running statistics_example (parallel bootstrapping) ---"
./bin/statistics_example
echo ""

