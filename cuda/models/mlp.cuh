// Fully connected network trained with AdamW on the GPU.
// Layout is feat to 64 to 32 to K with ReLU hidden layers and a softmax output.
// Activations and deltas for the whole dataset stay in device memory.
#pragma once

#include <cmath>
#include <initializer_list>
#include <vector>

#include "../common/cuda_utils.cuh"
#include "../common/rng.hpp"
#include "../common/timer.hpp"
#include "../kernels/device_math.cuh"
#include "../kernels/reduce.cuh"
#include "../optimizers/adamw.cuh"

// One thread per output activation of a dense layer.
// act_type 1 applies ReLU and 0 keeps the value linear.
__global__ void kernel_layer_fwd(
        const float* __restrict__ A_in,
        const float* __restrict__ W,
        const float* __restrict__ b,
        float* __restrict__ Z,
        float* __restrict__ A_out,
        int n, int in_dim, int out_dim, int act_type) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=n*out_dim;
    if(idx>=total) return;
    int i=idx/out_dim, o=idx%out_dim;
    float s=b[o];
    for(int j=0;j<in_dim;++j) s+=W[o*in_dim+j]*A_in[i*in_dim+j];
    Z[i*out_dim+o]=s;
    A_out[i*out_dim+o]=(act_type==1) ? d_relu(s) : s;
}

// One thread per sample. Softmax, cross entropy loss and the gradient at the logits.
__global__ void kernel_softmax_ce_bwd(
        const float* __restrict__ Z3,
        const int*   __restrict__ y,
        float* __restrict__ D3,
        float* __restrict__ loss_arr,
        int n, int K) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float mx=Z3[i*K];
    for(int k=1;k<K;++k) if(Z3[i*K+k]>mx) mx=Z3[i*K+k];
    float sum=0.f;
    for(int k=0;k<K;++k){ D3[i*K+k]=expf(Z3[i*K+k]-mx); sum+=D3[i*K+k]; }
    float inv_sum=1.f/sum;
    loss_arr[i]=0.f;
    for(int k=0;k<K;++k){
        D3[i*K+k]*=inv_sum;
        if(k==y[i]) loss_arr[i]=-logf(D3[i*K+k]+1e-9f);
    }
    D3[i*y[i]]-=1.f;
    for(int k=0;k<K;++k) D3[i*K+k]/=n;
}

// One thread per hidden unit and sample. Pushes the error back through the weights and the ReLU derivative.
__global__ void kernel_layer_bwd_delta(
        const float* __restrict__ W,
        const float* __restrict__ D_out,
        const float* __restrict__ Z_in,
        float* __restrict__ D_in,
        int n, int in_dim, int out_dim) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=n*in_dim;
    if(idx>=total) return;
    int i=idx/in_dim, j=idx%in_dim;
    float s=0.f;
    for(int o=0;o<out_dim;++o) s+=W[o*in_dim+j]*D_out[i*out_dim+o];
    D_in[i*in_dim+j]=s*d_relu_d(Z_in[i*in_dim+j]);
}

// 2D grid over the weight matrix. Sums the outer products over all samples.
__global__ void kernel_weight_grad(
        const float* __restrict__ A_in,
        const float* __restrict__ D_out,
        float* __restrict__ gW,
        int n, int in_dim, int out_dim) {
    int o=blockIdx.y*blockDim.y+threadIdx.y;
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(o>=out_dim||j>=in_dim) return;
    float s=0.f;
    for(int i=0;i<n;++i) s+=D_out[i*out_dim+o]*A_in[i*in_dim+j];
    gW[o*in_dim+j]=s;
}

// One thread per output unit. Sums the deltas over all samples.
__global__ void kernel_bias_grad(const float* __restrict__ D_out, float* __restrict__ gb, int n, int out_dim) {
    int o=blockIdx.x*blockDim.x+threadIdx.x;
    if(o>=out_dim) return;
    float s=0.f;
    for(int i=0;i<n;++i) s+=D_out[i*out_dim+o];
    gb[o]=s;
}

// Result of an MLP run.
struct MLPResult { double time_ms,final_loss,accuracy; int n_samples; };

// He initialization on the host, then full data training on the device.
// Every epoch is one AdamW step over all samples and batch_size is not used.
inline MLPResult mlp_adamw_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
                                 int n, int feat, int K, int epochs, int batch_size,
                                 float lr=1e-3f, float wd=1e-4f) {
    const int H1=64, H2=32;
    const float beta1=0.9f, beta2=0.999f, eps_adam=1e-8f;

    auto he=[&](int fan_in,int fan_out)->std::vector<float>{
        float s=sqrtf(2.f/fan_in);
        std::vector<float> v(fan_in*fan_out);
        for(auto& x:v) x=randnf()*s;
        return v;
    };
    auto zeros=[](int n_){ return std::vector<float>(n_,0.f); };

    auto hW1=he(feat,H1); auto hb1=zeros(H1);
    auto hW2=he(H1,H2);   auto hb2=zeros(H2);
    auto hW3=he(H2,K);    auto hb3=zeros(K);

    auto upload=[&](const std::vector<float>& v)->float*{
        float* p; CUDA_CHECK(cudaMalloc(&p,v.size()*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(p,v.data(),v.size()*sizeof(float),cudaMemcpyHostToDevice));
        return p;
    };
    auto gpu_zeros=[&](int sz)->float*{
        float* p; CUDA_CHECK(cudaMalloc(&p,sz*sizeof(float)));
        CUDA_CHECK(cudaMemset(p,0,sz*sizeof(float))); return p;
    };

    float *dX=upload(hX);
    int *dy; CUDA_CHECK(cudaMalloc(&dy,n*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dy,hy.data(),n*sizeof(int),cudaMemcpyHostToDevice));

    float *dW1=upload(hW1),*db1=upload(hb1);
    float *dW2=upload(hW2),*db2=upload(hb2);
    float *dW3=upload(hW3),*db3=upload(hb3);
    float *mW1=gpu_zeros(H1*feat),*vW1=gpu_zeros(H1*feat);
    float *mb1=gpu_zeros(H1),     *vb1=gpu_zeros(H1);
    float *mW2=gpu_zeros(H2*H1),  *vW2=gpu_zeros(H2*H1);
    float *mb2=gpu_zeros(H2),     *vb2=gpu_zeros(H2);
    float *mW3=gpu_zeros(K*H2),   *vW3=gpu_zeros(K*H2);
    float *mb3=gpu_zeros(K),      *vb3=gpu_zeros(K);
    float *dZ1=gpu_zeros(n*H1),*dA1=gpu_zeros(n*H1);
    float *dZ2=gpu_zeros(n*H2),*dA2=gpu_zeros(n*H2);
    float *dZ3=gpu_zeros(n*K);
    float *dD3=gpu_zeros(n*K),*dD2=gpu_zeros(n*H2),*dD1=gpu_zeros(n*H1);
    float *dgW1=gpu_zeros(H1*feat),*dgb1=gpu_zeros(H1);
    float *dgW2=gpu_zeros(H2*H1),  *dgb2=gpu_zeros(H2);
    float *dgW3=gpu_zeros(K*H2),   *dgb3=gpu_zeros(K);
    float *d_loss_arr=gpu_zeros(n);

    int blk=256;
    auto t0=Clock::now();
    double loss=0.0;
    int t=0;

    for(int ep=0;ep<epochs;++ep){
        ++t;
        float bc1=1.f-powf(beta1,(float)t);
        float bc2=1.f-powf(beta2,(float)t);

        // forward
        int tot1=n*H1; int g1=(tot1+blk-1)/blk;
        kernel_layer_fwd<<<g1,blk>>>(dX,dW1,db1,dZ1,dA1,n,feat,H1,1);
        int tot2=n*H2; int g2=(tot2+blk-1)/blk;
        kernel_layer_fwd<<<g2,blk>>>(dA1,dW2,db2,dZ2,dA2,n,H1,H2,1);
        int tot3=n*K; int g3=(tot3+blk-1)/blk;
        kernel_layer_fwd<<<g3,blk>>>(dA2,dW3,db3,dZ3,dZ3,n,H2,K,0);

        // backward
        kernel_softmax_ce_bwd<<<(n+blk-1)/blk,blk>>>(dZ3,dy,dD3,d_loss_arr,n,K);

        {dim3 b2(16,16), g2d((H2+15)/16,(K+15)/16);
         kernel_weight_grad<<<g2d,b2>>>(dA2,dD3,dgW3,n,H2,K);}
        kernel_bias_grad<<<(K+blk-1)/blk,blk>>>(dD3,dgb3,n,K);

        int totD2=n*H2; int gD2=(totD2+blk-1)/blk;
        kernel_layer_bwd_delta<<<gD2,blk>>>(dW3,dD3,dZ2,dD2,n,H2,K);

        {dim3 b2(16,16), g2d((H1+15)/16,(H2+15)/16);
         kernel_weight_grad<<<g2d,b2>>>(dA1,dD2,dgW2,n,H1,H2);}
        kernel_bias_grad<<<(H2+blk-1)/blk,blk>>>(dD2,dgb2,n,H2);

        int totD1=n*H1; int gD1=(totD1+blk-1)/blk;
        kernel_layer_bwd_delta<<<gD1,blk>>>(dW2,dD2,dZ1,dD1,n,H1,H2);

        {dim3 b2(16,16), g2d((feat+15)/16,(H1+15)/16);
         kernel_weight_grad<<<g2d,b2>>>(dX,dD1,dgW1,n,feat,H1);}
        kernel_bias_grad<<<(H1+blk-1)/blk,blk>>>(dD1,dgb1,n,H1);

        // adamw update all layers
        auto upd=[&](float* w, float* mw, float* vw, const float* gw, int sz, bool apply_wd){
            kernel_adamw_update<<<(sz+blk-1)/blk,blk>>>(
                w,mw,vw,gw,lr,beta1,beta2,eps_adam,
                apply_wd?wd:0.f, bc1,bc2,sz);
        };
        upd(dW1,mW1,vW1,dgW1,H1*feat,true);  upd(db1,mb1,vb1,dgb1,H1,false);
        upd(dW2,mW2,vW2,dgW2,H2*H1,true);    upd(db2,mb2,vb2,dgb2,H2,false);
        upd(dW3,mW3,vW3,dgW3,K*H2,true);     upd(db3,mb3,vb3,dgb3,K,false);

        if(ep==epochs-1) loss=gpu_sum(d_loss_arr,n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms=Ms(Clock::now()-t0).count();

    kernel_layer_fwd<<<(n*H1+blk-1)/blk,blk>>>(dX,dW1,db1,dZ1,dA1,n,feat,H1,1);
    kernel_layer_fwd<<<(n*H2+blk-1)/blk,blk>>>(dA1,dW2,db2,dZ2,dA2,n,H1,H2,1);
    kernel_layer_fwd<<<(n*K+blk-1)/blk,blk>>>(dA2,dW3,db3,dZ3,dZ3,n,H2,K,0);
    std::vector<float> h_z3(n*K);
    CUDA_CHECK(cudaMemcpy(h_z3.data(),dZ3,n*K*sizeof(float),cudaMemcpyDeviceToHost));
    int correct=0;
    for(int i=0;i<n;++i){
        int pred=0;
        for(int k=1;k<K;++k) if(h_z3[i*K+k]>h_z3[i*K+pred]) pred=k;
        if(pred==hy[i]) correct++;
    }

    auto frees=[](std::initializer_list<float*> ptrs){ for(auto p:ptrs) cudaFree(p); };
    frees({dX,dW1,db1,dW2,db2,dW3,db3});
    frees({mW1,vW1,mb1,vb1,mW2,vW2,mb2,vb2,mW3,vW3,mb3,vb3});
    frees({dZ1,dA1,dZ2,dA2,dZ3,dD3,dD2,dD1});
    frees({dgW1,dgb1,dgW2,dgb2,dgW3,dgb3,d_loss_arr});
    cudaFree(dy);
    return {ms,loss,100.0*correct/n,n};
}
