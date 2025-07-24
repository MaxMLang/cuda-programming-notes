/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
d * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// This file contains utility functions for CUDA program error checking and
// device management.

#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// CUDA Error Handling Wrapper
////////////////////////////////////////////////////////////////////////////////

#ifndef __DRIVER_TYPES_H__
#define __DRIVER_TYPES_H__
struct dim3 {
    unsigned int x, y, z;
};
#endif

// Define a macro for checking CUDA API calls for errors.
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorString(result), func);
    exit(EXIT_FAILURE);
  }
}

// This function is used to check for asynchronous kernel errors.
inline void getLastCudaError(const char *errorMessage) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "%s: %s\n", errorMessage, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


////////////////////////////////////////////////////////////////////////////////
// Device Selection
////////////////////////////////////////////////////////////////////////////////

inline int findCudaDevice(int argc, const char **argv) {
  int devID = 0;
  // If the command-line has a device number specified, use it
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "device=", 7) == 0) {
      devID = atoi(&argv[i][7]);
      break;
    }
  }

  int deviceCount = 0;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "Error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }
  if (devID >= deviceCount) {
    fprintf(stderr, "Error: device %d is not a valid CUDA device.\n", devID);
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaSetDevice(devID));
  
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
  printf("Using GPU %d: %s\n", devID, deviceProp.name);

  return devID;
}

#endif // HELPER_CUDA_H
