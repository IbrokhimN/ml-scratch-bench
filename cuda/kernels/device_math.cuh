// Device side activation functions.
#pragma once

#include <cuda_runtime.h>

__device__ inline float d_sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
__device__ inline float d_relu(float x)    { return x > 0.f ? x : 0.f; }
__device__ inline float d_relu_d(float x)  { return x > 0.f ? 1.f : 0.f; }
