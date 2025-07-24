# CUDA Programming Course Practicals

![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)

Course materials from the CUDA Programming on NVIDIA GPUs course taught by Prof. Mike Giles and Prof. Wes Armour at the University of Oxford.

**Course Website**: https://people.maths.ox.ac.uk/gilesm/cuda/
**Capstone Project**: https://github.com/MaxMLang/cuda-gis-smoothing/

## Overview

This repository contains my work on the practical exercises from the CUDA programming course. The code includes:
- Original examples from the course
- My own implementations and optimizations
- Modified versions with performance improvements
- A capstone project applying multiple concepts

## Practical Structure

### Core Practicals (1-12)
- **prac1**: Hello world examples - kernel launching, data transfer, error checking
- **prac2**: Monte Carlo simulation using cuRAND - constant memory, random generation, bandwidth optimization
- **prac3**: 3D Laplace solver - thread block optimization, memory layout, profiling
- **prac4**: Reduction operations - shared memory, synchronization, shuffles, atomics
- **prac5**: Tensor Cores and cuBLAS - library usage and optimization
- **prac6**: C++ integration - templates, libraries, mixed compilation
- **prac7**: Tri-diagonal equations - specialized algorithms
- **prac8**: Scan operations - recurrence equations and parallel algorithms
- **prac9**: Pattern matching - string processing on GPU
- **prac10**: Auto-tuning - performance optimization techniques
- **prac11**: Streams and OpenMP - asynchronous processing and multithreading
- **prac12**: Advanced streaming - computation/communication overlap

## Key Learning Areas

- Memory management (global, shared, constant, texture)
- Thread organization and synchronization
- Performance optimization and profiling
- CUDA libraries (cuBLAS, cuRAND, cuFFT)
- Multi-GPU programming
- Debugging and error handling

## Course Prerequisites

- C/C++ programming experience
- No prior parallel computing knowledge required
- Access to NVIDIA GPU hardware

## Building and Running

Each practical directory contains:
- Source files (.cu, .cpp)
- Makefile for compilation
- Shell script for execution
- Answer files documenting solutions and insights

## Capstone Project

A satellite data smoothing case study that integrates multiple concepts from the practicals, demonstrating real-world application of CUDA programming techniques.

## Notes

- Some implementations are direct from course materials
- Others include my own optimizations and modifications
- Performance improvements and alternative approaches documented in answer files
- All work completed as part of the course curriculum
