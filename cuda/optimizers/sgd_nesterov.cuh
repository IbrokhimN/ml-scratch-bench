// SGD with Nesterov momentum for binary logistic regression on the GPU.
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

// One thread per parameter. Updates the velocity and then the weights.
__global__ void kernel_nesterov_update(
        float* __restrict__ w,
        float* __restrict__ vel,
        const float* __restrict__ g,
        float lr, float mu, int feat) {
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=feat) return;
    vel[j] = mu*vel[j] - lr*g[j];
    w[j]  += vel[j];
}

// Writes the lookahead point w plus mu times vel into w_la.
__global__ void kernel_lookahead(const float* w, const float* vel,
                                   float* w_la, float mu, int feat) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<feat) w_la[j]=w[j]+mu*vel[j];
}

// Gradients are taken at the lookahead point and the learning rate follows a cosine decay.
// Every step uses all samples, so one epoch is one step and batch_size is not used.
// The bias update runs on the host from a reduced gradient.
inline OptResult sgd_nesterov_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
                                     int n, int feat, int epochs, int batch_size,
                                     float lr=0.01f, float mu=0.9f) {
    float *dX; int *dy;
    CUDA_CHECK(cudaMalloc(&dX, n*feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dX,hX.data(),n*feat*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy,hy.data(),n*sizeof(int),cudaMemcpyHostToDevice));

    float *dw,*dvel,*dw_la,*d_err,*d_bce,*d_gw;
    CUDA_CHECK(cudaMalloc(&dw,    feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dvel,  feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw_la, feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_err, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bce, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw,  feat*sizeof(float)));
    CUDA_CHECK(cudaMemset(dw,0,feat*sizeof(float)));
    CUDA_CHECK(cudaMemset(dvel,0,feat*sizeof(float)));

    int grid_n=(n+255)/256, grid_f=(feat+255)/256;
    float h_bias=0.f,h_vel_b=0.f;
    double loss=0.0;
    auto t0=Clock::now();

    for(int ep=0;ep<epochs;++ep){
        float lr_t=lr*0.5f*(1.f+cosf((float)M_PI*ep/epochs));
        lr_t=fmaxf(lr_t, lr*0.01f);

        kernel_lookahead<<<grid_f,256>>>(dw,dvel,dw_la,mu,feat);
        float h_bias_la=h_bias+mu*h_vel_b;

        kernel_logistic_forward<<<grid_n,256>>>(dX,dw_la,h_bias_la,dy,d_err,d_bce,n,feat);
        kernel_grad_weights<<<grid_f,256>>>(dX,d_err,d_gw,n,feat);

        float sum_err=gpu_sum(d_err,n)/n;
        h_vel_b=mu*h_vel_b-lr_t*sum_err;
        h_bias+=h_vel_b;

        kernel_nesterov_update<<<grid_f,256>>>(dw,dvel,d_gw,lr_t,mu,feat);
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
    cudaFree(dX); cudaFree(dy); cudaFree(dw); cudaFree(dvel); cudaFree(dw_la);
    cudaFree(d_err); cudaFree(d_bce); cudaFree(d_gw);
    return {ms,loss,100.0*correct/n,n,"SGD_Nesterov"};
}
