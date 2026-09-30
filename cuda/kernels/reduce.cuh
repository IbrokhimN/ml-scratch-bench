// Parallel sum reduction used by almost every algorithm.
#pragma once

#include "../common/cuda_utils.cuh"

// Tree reduction in shared memory. Each block adds its partial sum into out with an atomic.
__global__ void kernel_sum_reduce(const float* __restrict__ in, float* __restrict__ out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x * 2 + tid;
    sdata[tid] = 0.f;
    if (i < n)              sdata[tid]  = in[i];
    if (i + blockDim.x < n) sdata[tid] += in[i + blockDim.x];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// Returns the sum of a device array on the host.
// Allocates a scratch scalar and synchronizes on every call.
static float gpu_sum(const float* d_arr, int n, int blk=256) {
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    int grid = (n + blk*2 - 1) / (blk*2);
    kernel_sum_reduce<<<grid, blk, blk*sizeof(float)>>>(d_arr, d_out, n);
    float h_out;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    return h_out;
}
