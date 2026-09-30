// Kernels shared by every logistic regression optimizer.
// One forward kernel produces the errors and one kernel turns them into weight gradients.
#pragma once

#include "../kernels/device_math.cuh"

// One thread per sample. Computes the prediction error and the BCE loss.
__global__ void kernel_logistic_forward(
        const float* __restrict__ X,
        const float* __restrict__ w,
        float bias,
        const int*   __restrict__ y,
        float* __restrict__ err,
        float* __restrict__ bce,
        int n, int feat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z = bias;
    for (int j = 0; j < feat; ++j) z += w[j] * X[i*feat+j];
    float p   = d_sigmoid(z);
    float yi  = (float)y[i];
    err[i]    = p - yi;
    bce[i]    = -(yi*logf(p+1e-9f) + (1.f-yi)*logf(1.f-p+1e-9f));
}

// One thread per feature. Averages err times X over all samples.
__global__ void kernel_grad_weights(
        const float* __restrict__ X,
        const float* __restrict__ err,
        float* __restrict__ gw,
        int n, int feat) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= feat) return;
    float g = 0.f;
    for (int i = 0; i < n; ++i) g += err[i] * X[i*feat+j];
    gw[j] = g / n;
}
