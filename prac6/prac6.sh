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
#SBATCH --job-name=prac6

# use our reservation
#SBATCH --reservation=cuda2025

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Cleaning and building projects... ---"
make clean
make all
echo "--- Build complete. ---"
echo

# --- Run Original Practicals ---
echo "--- Running prac6 ---"
./bin/prac6
echo

echo "--- Running prac6a ---"
./bin/prac6a
echo

echo "--- Running prac6b ---"
./bin/prac6b
echo

echo "--- Running prac6c ---"
./bin/prac6c
echo

# --- Run New Statistics Demo ---
echo "--- Running stats demo ---"
./bin/stats
echo