// Nadam for binary logistic regression on the GPU.
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

// One thread per parameter. Adam update with the Nesterov correction folded in.
__global__ void kernel_nadam_update(
        float* __restrict__ w,
        float* __restrict__ m,
        float* __restrict__ v,
        const float* __restrict__ g,
        float lr, float beta1, float beta2, float eps,
        float bc1, float bc2, float bc1_next,
        int feat) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= feat) return;
    float gj = g[j];
    m[j] = beta1*m[j] + (1.f-beta1)*gj;
    v[j] = beta2*v[j] + (1.f-beta2)*gj*gj;
    float v_hat     = v[j] / bc2;
    float nadam_dir = (beta1 * m[j]/bc1_next) + ((1.f-beta1)*gj/bc1);
    w[j] -= lr * nadam_dir / (sqrtf(v_hat) + eps);
}

// Every step uses all samples, so one epoch is one step and batch_size is not used.
// The bias update runs on the host from a reduced gradient.
inline OptResult nadam_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
                             int n, int feat, int epochs, int batch_size,
                             float lr=1e-3f, float beta1=0.9f, float beta2=0.999f, float eps=1e-8f) {
    float *dX; int *dy;
    CUDA_CHECK(cudaMalloc(&dX, n*feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), n*feat*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), n*sizeof(int),        cudaMemcpyHostToDevice));

    float *dw,*dm,*dv,*d_err,*d_bce,*d_gw;
    CUDA_CHECK(cudaMalloc(&dw,    feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dm,    feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dv,    feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_err, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bce, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw,  feat*sizeof(float)));
    CUDA_CHECK(cudaMemset(dw,0,feat*sizeof(float)));
    CUDA_CHECK(cudaMemset(dm,0,feat*sizeof(float)));
    CUDA_CHECK(cudaMemset(dv,0,feat*sizeof(float)));

    int grid_n=(n+255)/256, grid_f=(feat+255)/256;
    float h_bias=0.f; int t=0; double loss=0.0;
    auto t0=Clock::now();

    for(int ep=0;ep<epochs;++ep){
        ++t;
        float bc1=1.f-powf(beta1,(float)t);
        float bc2=1.f-powf(beta2,(float)t);
        float bc1_next=1.f-powf(beta1,(float)(t+1));

        kernel_logistic_forward<<<grid_n,256>>>(dX,dw,h_bias,dy,d_err,d_bce,n,feat);
        kernel_grad_weights<<<grid_f,256>>>(dX,d_err,d_gw,n,feat);

        float sum_err=gpu_sum(d_err,n);
        h_bias-=lr*sum_err/n;

        kernel_nadam_update<<<grid_f,256>>>(dw,dm,dv,d_gw,lr,beta1,beta2,eps,bc1,bc2,bc1_next,feat);

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
    cudaFree(dX); cudaFree(dy); cudaFree(dw); cudaFree(dm); cudaFree(dv);
    cudaFree(d_err); cudaFree(d_bce); cudaFree(d_gw);
    return {ms,loss,100.0*correct/n,n,"Nadam"};
}
