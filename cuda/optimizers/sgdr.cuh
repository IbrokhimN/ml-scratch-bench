// SGD with cosine warm restarts for binary logistic regression on the GPU.
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

// One thread per parameter. Plain gradient descent step.
__global__ void kernel_sgd_update(float* __restrict__ w, const float* __restrict__ g,
                                   float lr, int feat) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<feat) w[j]-=lr*g[j];
}

// The first cycle lasts T0 epochs and every later cycle is T_mult times longer.
// Every step uses all samples, so one epoch is one step and batch_size is not used.
// The bias update runs on the host from a reduced gradient.
inline OptResult sgdr_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
                            int n, int feat, int epochs, int batch_size,
                            float lr_max=0.05f, float lr_min=1e-5f, int T0=10, int T_mult=2) {
    float *dX; int *dy;
    CUDA_CHECK(cudaMalloc(&dX, n*feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dX,hX.data(),n*feat*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy,hy.data(),n*sizeof(int),cudaMemcpyHostToDevice));

    float *dw,*d_err,*d_bce,*d_gw;
    CUDA_CHECK(cudaMalloc(&dw,    feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_err, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bce, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw,  feat*sizeof(float)));
    CUDA_CHECK(cudaMemset(dw,0,feat*sizeof(float)));

    int grid_n=(n+255)/256, grid_f=(feat+255)/256;
    float h_bias=0.f;
    int T_cur=0, T_i=T0;
    double loss=0.0;
    auto t0=Clock::now();

    for(int ep=0;ep<epochs;++ep){
        float lr=lr_min+0.5f*(lr_max-lr_min)*(1.f+cosf((float)M_PI*T_cur/T_i));
        if(++T_cur>=T_i){ T_cur=0; T_i*=T_mult; }

        kernel_logistic_forward<<<grid_n,256>>>(dX,dw,h_bias,dy,d_err,d_bce,n,feat);
        kernel_grad_weights<<<grid_f,256>>>(dX,d_err,d_gw,n,feat);

        float sum_err=gpu_sum(d_err,n)/n;
        h_bias-=lr*sum_err;

        kernel_sgd_update<<<grid_f,256>>>(dw,d_gw,lr,feat);
        if(ep==epochs-1) loss=gpu_sum(d_bce,n)/n;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms=Ms(Clock::now()-t0).count();

    std::vector<float> hw(feat);
    CUDA_CHECK(cudaMemcpy(hw.data(),dw,feat*sizeof(float),cudaMemcpyDeviceToHost));
    int correct=0;
    for(int i=0;i<n;++i){
        float z=h_bias; for(int j=0;j<feat;++j) z+=hw[j]*hX[i*feat+j];
        if((z>0.f?1:0)==hy[i]) correct++;
    }
    cudaFree(dX); cudaFree(dy); cudaFree(dw); cudaFree(d_err); cudaFree(d_bce); cudaFree(d_gw);
    return {ms,loss,100.0*correct/n,n,"SGDR"};
}
