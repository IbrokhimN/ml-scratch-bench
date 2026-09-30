// RMSProp with momentum for binary logistic regression on the GPU.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "../common/cuda_utils.cuh"
#include "../common/rng.hpp"
#include "../common/timer.hpp"
#include "../kernels/reduce.cuh"
#include "logistic_kernels.cuh"
#include "opt_result.hpp"

// One thread per parameter. Running average of squared gradients and a momentum step.
__global__ void kernel_rmsprop_update(
        float* __restrict__ w,
        float* __restrict__ eg2,
        float* __restrict__ delta,
        const float* __restrict__ g,
        float lr, float rho, float eps, float momentum,
        int feat) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j>=feat) return;
    float gj = g[j];
    eg2[j]   = rho*eg2[j] + (1.f-rho)*gj*gj;
    delta[j] = momentum*delta[j] - lr*gj/(sqrtf(eg2[j])+eps);
    w[j]    += delta[j];
}

// Every step uses all samples, so one epoch is one step and batch_size is not used.
// The bias update runs on the host from a reduced gradient.
inline OptResult rmsprop_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
                               int n, int feat, int epochs, int batch_size,
                               float lr=1e-3f, float rho=0.9f, float eps=1e-8f, float mom=0.9f) {
    float *dX; int *dy;
    CUDA_CHECK(cudaMalloc(&dX, n*feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dX,hX.data(),n*feat*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy,hy.data(),n*sizeof(int),cudaMemcpyHostToDevice));

    float *dw,*deg2,*ddelta,*d_err,*d_bce,*d_gw;
    CUDA_CHECK(cudaMalloc(&dw,    feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&deg2,  feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ddelta,feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_err, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bce, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw,  feat*sizeof(float)));
    CUDA_CHECK(cudaMemset(dw,0,feat*sizeof(float)));
    CUDA_CHECK(cudaMemset(deg2,0,feat*sizeof(float)));
    CUDA_CHECK(cudaMemset(ddelta,0,feat*sizeof(float)));

    int grid_n=(n+255)/256, grid_f=(feat+255)/256;
    float h_bias=0.f,eg2_b=0.f,delta_b=0.f;
    double loss=0.0;
    auto t0=Clock::now();

    for(int ep=0;ep<epochs;++ep){
        kernel_logistic_forward<<<grid_n,256>>>(dX,dw,h_bias,dy,d_err,d_bce,n,feat);
        kernel_grad_weights<<<grid_f,256>>>(dX,d_err,d_gw,n,feat);

        float sum_err=gpu_sum(d_err,n)/n;
        eg2_b   = rho*eg2_b   + (1.f-rho)*sum_err*sum_err;
        delta_b = mom*delta_b  - lr*sum_err/(sqrtf(eg2_b)+eps);
        h_bias += delta_b;

        kernel_rmsprop_update<<<grid_f,256>>>(dw,deg2,ddelta,d_gw,lr,rho,eps,mom,feat);
        if(ep==epochs-1) loss=gpu_sum(d_bce,n)/n;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms=Ms(Clock::now()-t0).count();

    std::vector<float> hw(feat);
    CUDA_CHECK(cudaMemcpy(hw.data(),dw,feat*sizeof(float),cudaMemcpyDeviceToHost));
    int correct=0;
    for(int i=0;i<n;++i){
        float z=h_bias;
        for(int j=0;j<feat;++j) z+=hw[j]*hX[i*feat+j];
        if((z>0.f?1:0)==hy[i]) correct++;
    }
    cudaFree(dX); cudaFree(dy); cudaFree(dw); cudaFree(deg2); cudaFree(ddelta);
    cudaFree(d_err); cudaFree(d_bce); cudaFree(d_gw);
    return {ms,loss,100.0*correct/n,n,"RMSProp"};
}
