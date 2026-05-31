// ml_cuda.cu — ML algos on GPU, no cuDNN/cuML, everything from scratch
// compile: nvcc -std=c++17 -O3 -arch=sm_75 -o ml_cuda ml_cuda.cu -lm

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <cassert>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static std::mt19937_64 G_RNG(0xDEADBEEFCAFE1234ULL);
inline float randf()  { return std::uniform_real_distribution<float>(0.f,1.f)(G_RNG); }
inline float randnf() { return std::normal_distribution<float>(0.f,1.f)(G_RNG); }
inline int   randi(int lo, int hi) { return std::uniform_int_distribution<int>(lo,hi-1)(G_RNG); }

void gen_multiclass(std::vector<float>& X, std::vector<int>& y, int n, int feat, int K) {
    X.resize(n*feat); y.resize(n);
    std::vector<float> centres(K*feat);
    for(auto& v:centres) v=randnf()*2.f;
    for(int i=0;i<n;++i){
        int cls=randi(0,K); y[i]=cls;
        for(int j=0;j<feat;++j) X[i*feat+j]=centres[cls*feat+j]+randnf()*0.8f;
    }
}

// parallel sum reduction, block reduces to single value via atomicAdd
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

__device__ inline float d_sigmoid(float x) { return 1.f / (1.f + expf(-x)); }
__device__ inline float d_relu(float x)    { return x > 0.f ? x : 0.f; }
__device__ inline float d_relu_d(float x)  { return x > 0.f ? 1.f : 0.f; }

// logistic forward: pred = sigmoid(X*w+b), err = pred-y, bce loss
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

// grad weights: gw[j] = (1/n) sum_i err[i] * X[i,j]
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

// adamw
__global__ void kernel_adamw_update(
        float* __restrict__ w,
        float* __restrict__ m,
        float* __restrict__ v,
        const float* __restrict__ g,
        float lr, float beta1, float beta2, float eps, float wd,
        float bc1, float bc2,
        int feat) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= feat) return;
    float gj = g[j];
    m[j] = beta1*m[j] + (1.f-beta1)*gj;
    v[j] = beta2*v[j] + (1.f-beta2)*gj*gj;
    float m_hat = m[j] / bc1;
    float v_hat = v[j] / bc2;
    w[j] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * w[j]);
}

struct OptResult { double time_ms, final_loss, accuracy; int n_samples; std::string tag; };

OptResult adamw_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
                      int n, int feat, int epochs, int batch_size,
                      float lr=1e-3f, float beta1=0.9f, float beta2=0.999f,
                      float eps=1e-8f, float wd=1e-2f) {
    float *dX; int *dy;
    CUDA_CHECK(cudaMalloc(&dX, n*feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), n*feat*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), n*sizeof(int),        cudaMemcpyHostToDevice));

    float *dw, *dm, *dv, *d_err, *d_bce, *d_gw;
    CUDA_CHECK(cudaMalloc(&dw,    feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dm,    feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dv,    feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_err, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bce, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw,  feat*sizeof(float)));
    CUDA_CHECK(cudaMemset(dw, 0, feat*sizeof(float)));
    CUDA_CHECK(cudaMemset(dm, 0, feat*sizeof(float)));
    CUDA_CHECK(cudaMemset(dv, 0, feat*sizeof(float)));

    int blk_n = 256, blk_f = 256;
    int grid_n = (n+blk_n-1)/blk_n, grid_f = (feat+blk_f-1)/blk_f;

    float h_bias=0.f;
    int t=0;
    double loss=0.0;
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);

    auto t0 = Clock::now();

    for (int ep=0; ep<epochs; ++ep) {
        std::shuffle(idx.begin(),idx.end(),G_RNG);
        loss=0.0;
        for (int start=0; start<n; start+=batch_size) {
            int end=std::min(start+batch_size,n), bs=end-start;
            ++t;
            float bc1=1.f-powf(beta1,(float)t);
            float bc2=1.f-powf(beta2,(float)t);

            kernel_logistic_forward<<<grid_n,blk_n>>>(dX, dw, h_bias, dy, d_err, d_bce, n, feat);
            kernel_grad_weights<<<grid_f,blk_f>>>(dX, d_err, d_gw, n, feat);

            float sum_err = gpu_sum(d_err, n);
            h_bias -= lr * sum_err / n;

            kernel_adamw_update<<<grid_f,blk_f>>>(dw, dm, dv, d_gw, lr, beta1, beta2, eps, wd, bc1, bc2, feat);

            if (ep==epochs-1) loss += gpu_sum(d_bce, n) / n;
            break;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms = Ms(Clock::now()-t0).count();

    std::vector<float> hw(feat);
    CUDA_CHECK(cudaMemcpy(hw.data(), dw, feat*sizeof(float), cudaMemcpyDeviceToHost));
    int correct=0;
    for(int i=0;i<n;++i){
        float z=h_bias;
        for(int j=0;j<feat;++j) z+=hw[j]*hX[i*feat+j];
        if((z>0.f?1:0)==hy[i]) correct++;
    }

    cudaFree(dX); cudaFree(dy); cudaFree(dw); cudaFree(dm); cudaFree(dv);
    cudaFree(d_err); cudaFree(d_bce); cudaFree(d_gw);
    return {ms, loss, 100.0*correct/n, n, "AdamW"};
}

// nadam — nesterov correction in update kernel
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

OptResult nadam_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
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

// rmsprop
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

OptResult rmsprop_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
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

// sgd nesterov
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

__global__ void kernel_lookahead(const float* w, const float* vel,
                                   float* w_la, float mu, int feat) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<feat) w_la[j]=w[j]+mu*vel[j];
}

OptResult sgd_nesterov_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
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

// sgdr
__global__ void kernel_sgd_update(float* __restrict__ w, const float* __restrict__ g,
                                   float lr, int feat) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<feat) w[j]-=lr*g[j];
}

OptResult sgdr_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
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

// lbfgs — forward pass on GPU, two-loop recursion on GPU via reductions
__global__ void kernel_bce_grad(
        const float* __restrict__ X,
        const float* __restrict__ theta,
        const int*   __restrict__ y,
        float* __restrict__ loss_arr,
        float* __restrict__ grad,
        int n, int feat) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float z=theta[feat];
    for(int j=0;j<feat;++j) z+=theta[j]*X[i*feat+j];
    float p=d_sigmoid(z), yi=(float)y[i];
    loss_arr[i]=-(yi*logf(p+1e-9f)+(1.f-yi)*logf(1.f-p+1e-9f));
    float dz=(p-yi)/n;
    for(int j=0;j<feat;++j) atomicAdd(&grad[j], dz*X[i*feat+j]);
    atomicAdd(&grad[feat], dz);
}

__global__ void kernel_dot(const float* a, const float* b, float* out, int d) {
    extern __shared__ float sdata[];
    int tid=threadIdx.x;
    sdata[tid]=0.f;
    for(int i=tid;i<d;i+=blockDim.x) sdata[tid]+=a[i]*b[i];
    __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s) sdata[tid]+=sdata[tid+s];
        __syncthreads();
    }
    if(tid==0) *out=sdata[0];
}

__global__ void kernel_axpy(float* r, const float* v, float alpha, int d){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<d) r[j]+=alpha*v[j];
}

__global__ void kernel_axpy_neg(float* r, const float* v, float alpha, int d){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<d) r[j]-=alpha*v[j];
}

__global__ void kernel_scale(float* out, const float* in, float s, int d){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<d) out[j]=in[j]*s;
}

struct LBFGSResult { double time_ms,final_loss,accuracy; int n_samples; };

LBFGSResult lbfgs_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
                        int n, int feat, int max_iter=100, int M=10) {
    int d=feat+1;

    float *dX; int *dy;
    CUDA_CHECK(cudaMalloc(&dX, n*feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dX,hX.data(),n*feat*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy,hy.data(),n*sizeof(int),cudaMemcpyHostToDevice));

    float *d_theta,*d_grad,*d_loss_arr,*d_dot_out;
    CUDA_CHECK(cudaMalloc(&d_theta,   d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad,    d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss_arr,n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dot_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_theta,0,d*sizeof(float)));

    std::vector<float*> d_s(M), d_y_h(M);
    for(int i=0;i<M;++i){
        CUDA_CHECK(cudaMalloc(&d_s[i],  d*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y_h[i],d*sizeof(float)));
    }
    std::vector<float> rho_hist(M,0.f), alpha_arr(M,0.f);
    int history_len=0, head=0;

    float *d_q, *d_r;
    CUDA_CHECK(cudaMalloc(&d_q, d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r, d*sizeof(float)));

    int grid_n=(n+255)/256, grid_d=(d+255)/256;

    auto gpu_dot=[&](const float* a, const float* b)->float{
        CUDA_CHECK(cudaMemset(d_dot_out,0,sizeof(float)));
        kernel_dot<<<1,256,256*sizeof(float)>>>(a,b,d_dot_out,d);
        float h; CUDA_CHECK(cudaMemcpy(&h,d_dot_out,sizeof(float),cudaMemcpyDeviceToHost));
        return h;
    };

    auto eval=[&](float* theta, float* grad)->double{
        CUDA_CHECK(cudaMemset(grad,0,d*sizeof(float)));
        CUDA_CHECK(cudaMemset(d_loss_arr,0,n*sizeof(float)));
        kernel_bce_grad<<<grid_n,256>>>(dX,theta,dy,d_loss_arr,grad,n,feat);
        CUDA_CHECK(cudaDeviceSynchronize());
        return (double)gpu_sum(d_loss_arr,n);
    };

    auto t0=Clock::now();
    double loss=eval(d_theta,d_grad)/n;

    float *d_theta_new,*d_grad_new;
    CUDA_CHECK(cudaMalloc(&d_theta_new,d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_new, d*sizeof(float)));

    for(int iter=0;iter<max_iter;++iter){
        // two-loop recursion
        CUDA_CHECK(cudaMemcpy(d_q,d_grad,d*sizeof(float),cudaMemcpyDeviceToDevice));

        for(int i=history_len-1;i>=0;--i){
            int idx=(head-1-i+2*M)%M;
            float ai=rho_hist[idx]*gpu_dot(d_s[idx],d_q);
            alpha_arr[i]=ai;
            kernel_axpy_neg<<<grid_d,256>>>(d_q,d_y_h[idx],ai,d);
        }

        float gamma=1.f;
        if(history_len>0){
            int last=(head-1+M)%M;
            float sy=gpu_dot(d_s[last],d_y_h[last]);
            float yy=gpu_dot(d_y_h[last],d_y_h[last]);
            if(yy>1e-10f) gamma=sy/yy;
        }
        kernel_scale<<<grid_d,256>>>(d_r,d_q,gamma,d);

        for(int i=0;i<history_len;++i){
            int idx=(head-history_len+i+2*M)%M;
            float beta=rho_hist[idx]*gpu_dot(d_y_h[idx],d_r);
            kernel_axpy<<<grid_d,256>>>(d_r,d_s[idx],alpha_arr[i]-beta,d);
        }

        kernel_scale<<<grid_d,256>>>(d_q,d_r,-1.f,d);

        // armijo line search
        float step=1.f;
        float pg=gpu_dot(d_q,d_grad);
        float c1=1e-4f;
        double loss_new=loss+1.0;
        for(int ls=0;ls<30&&step>1e-12f;++ls){
            CUDA_CHECK(cudaMemcpy(d_theta_new,d_theta,d*sizeof(float),cudaMemcpyDeviceToDevice));
            kernel_axpy<<<grid_d,256>>>(d_theta_new,d_q,step,d);
            loss_new=eval(d_theta_new,d_grad_new)/n;
            if(loss_new <= loss + (double)(c1*step*pg)) break;
            step*=0.5f;
        }

        int hi=head;
        CUDA_CHECK(cudaMemcpy(d_s[hi],d_theta_new,d*sizeof(float),cudaMemcpyDeviceToDevice));
        kernel_axpy_neg<<<grid_d,256>>>(d_s[hi],d_theta,1.f,d);
        CUDA_CHECK(cudaMemcpy(d_y_h[hi],d_grad_new,d*sizeof(float),cudaMemcpyDeviceToDevice));
        kernel_axpy_neg<<<grid_d,256>>>(d_y_h[hi],d_grad,1.f,d);

        float sy=gpu_dot(d_s[hi],d_y_h[hi]);
        rho_hist[hi]=(sy>1e-10f)?1.f/sy:0.f;
        head=(head+1)%M;
        if(history_len<M) history_len++;

        CUDA_CHECK(cudaMemcpy(d_theta,d_theta_new,d*sizeof(float),cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad, d_grad_new, d*sizeof(float),cudaMemcpyDeviceToDevice));
        loss=loss_new;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double ms=Ms(Clock::now()-t0).count();

    std::vector<float> h_theta(d);
    CUDA_CHECK(cudaMemcpy(h_theta.data(),d_theta,d*sizeof(float),cudaMemcpyDeviceToHost));
    int correct=0;
    for(int i=0;i<n;++i){
        float z=h_theta[feat]; for(int j=0;j<feat;++j) z+=h_theta[j]*hX[i*feat+j];
        if((z>0.f?1:0)==hy[i]) correct++;
    }

    cudaFree(dX); cudaFree(dy); cudaFree(d_theta); cudaFree(d_grad);
    cudaFree(d_loss_arr); cudaFree(d_dot_out); cudaFree(d_q); cudaFree(d_r);
    cudaFree(d_theta_new); cudaFree(d_grad_new);
    for(int i=0;i<M;++i){ cudaFree(d_s[i]); cudaFree(d_y_h[i]); }
    return {ms,loss,100.0*correct/n,n};
}

// gmm em — parallel e-step and m-step kernels
__global__ void kernel_gmm_e_step(
        const float* __restrict__ X,
        const float* __restrict__ mu,
        const float* __restrict__ var,
        const float* __restrict__ pi,
        float* __restrict__ resp,
        float* __restrict__ log_lik_arr,
        int n, int feat, int K) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;

    const float LOG_2PI=1.8378770664093455f;
    float mx=-1e30f;
    float log_p[32];
    for(int k=0;k<K;++k){
        float lp=logf(pi[k]+1e-9f);
        for(int j=0;j<feat;++j){
            float d=X[i*feat+j]-mu[k*feat+j];
            lp+=-0.5f*(LOG_2PI+logf(var[k*feat+j]+1e-6f)+d*d/(var[k*feat+j]+1e-6f));
        }
        log_p[k]=lp;
        if(lp>mx) mx=lp;
    }
    float sum_exp=0.f;
    for(int k=0;k<K;++k){ resp[i*K+k]=expf(log_p[k]-mx); sum_exp+=resp[i*K+k]; }
    log_lik_arr[i]=logf(sum_exp)+mx;
    for(int k=0;k<K;++k) resp[i*K+k]/=sum_exp;
}

__global__ void kernel_gmm_m_nk(const float* __restrict__ resp, float* __restrict__ Nk, int n, int K) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    for(int k=0;k<K;++k) atomicAdd(&Nk[k], resp[i*K+k]);
}

__global__ void kernel_gmm_m_mu(
        const float* __restrict__ X, const float* __restrict__ resp,
        float* __restrict__ new_mu, int n, int feat, int K) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    for(int k=0;k<K;++k){
        float r=resp[i*K+k];
        for(int j=0;j<feat;++j) atomicAdd(&new_mu[k*feat+j], r*X[i*feat+j]);
    }
}

__global__ void kernel_gmm_divide_mu(float* __restrict__ mu, const float* __restrict__ Nk, int K, int feat) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=K) return;
    float nk=fmaxf(Nk[k],1e-6f);
    for(int j=0;j<feat;++j) mu[k*feat+j]/=nk;
}

__global__ void kernel_gmm_m_var(
        const float* __restrict__ X, const float* __restrict__ resp,
        const float* __restrict__ mu, float* __restrict__ new_var,
        int n, int feat, int K) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    for(int k=0;k<K;++k){
        float r=resp[i*K+k];
        for(int j=0;j<feat;++j){
            float d=X[i*feat+j]-mu[k*feat+j];
            atomicAdd(&new_var[k*feat+j], r*d*d);
        }
    }
}

__global__ void kernel_gmm_divide_var(float* __restrict__ var, const float* __restrict__ Nk, int K, int feat) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=K) return;
    float nk=fmaxf(Nk[k],1e-6f);
    for(int j=0;j<feat;++j) var[k*feat+j]=var[k*feat+j]/nk+1e-4f;
}

__global__ void kernel_gmm_pi(float* pi, const float* Nk, float n_inv, int K){
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k<K) pi[k]=fmaxf(Nk[k],1e-6f)*n_inv;
}

struct GMMResult { double time_ms,log_likelihood; int n_samples,K; };

GMMResult gmm_em_cuda(const std::vector<float>& hX, int n, int feat, int K, int max_iter=100) {
    // kmeans++ init on host
    std::vector<int> chosen;
    chosen.push_back(randi(0,n));
    for(int c=1;c<K;++c){
        std::vector<float> dist2(n,1e30f);
        for(int i=0;i<n;++i)
            for(int prev:chosen){
                float d=0.f;
                for(int j=0;j<feat;++j){float dd=hX[i*feat+j]-hX[prev*feat+j];d+=dd*dd;}
                dist2[i]=std::min(dist2[i],d);
            }
        float tot=0.f; for(float v:dist2) tot+=v;
        float r=randf()*tot,cum=0.f; int pick=n-1;
        for(int i=0;i<n;++i){cum+=dist2[i];if(cum>=r){pick=i;break;}}
        chosen.push_back(pick);
    }
    std::vector<float> h_mu(K*feat), h_var(K*feat,1.f), h_pi(K,1.f/K);
    for(int c=0;c<K;++c)
        for(int j=0;j<feat;++j) h_mu[c*feat+j]=hX[chosen[c]*feat+j];

    float *dX,*d_mu,*d_var,*d_pi,*d_resp,*d_ll,*d_Nk,*d_new_mu,*d_new_var;
    CUDA_CHECK(cudaMalloc(&dX,        n*feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mu,      K*feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var,     K*feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pi,      K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_resp,    n*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ll,      n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Nk,      K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_mu,  K*feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_var, K*feat*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX,   hX.data(),   n*feat*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, h_mu.data(), K*feat*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_var,h_var.data(),K*feat*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pi, h_pi.data(), K*sizeof(float),      cudaMemcpyHostToDevice));

    int grid_n=(n+255)/256, grid_k=(K+255)/256;
    auto t0=Clock::now();
    double log_lik=0.0;

    for(int iter=0;iter<max_iter;++iter){
        // e-step
        kernel_gmm_e_step<<<grid_n,256>>>(dX,d_mu,d_var,d_pi,d_resp,d_ll,n,feat,K);
        log_lik=(double)gpu_sum(d_ll,n)/n;

        // m-step
        CUDA_CHECK(cudaMemset(d_Nk,0,K*sizeof(float)));
        kernel_gmm_m_nk<<<grid_n,256>>>(d_resp,d_Nk,n,K);

        CUDA_CHECK(cudaMemset(d_new_mu,0,K*feat*sizeof(float)));
        kernel_gmm_m_mu<<<grid_n,256>>>(dX,d_resp,d_new_mu,n,feat,K);
        kernel_gmm_divide_mu<<<grid_k,256>>>(d_new_mu,d_Nk,K,feat);

        CUDA_CHECK(cudaMemset(d_new_var,0,K*feat*sizeof(float)));
        kernel_gmm_m_var<<<grid_n,256>>>(dX,d_resp,d_new_mu,d_new_var,n,feat,K);
        kernel_gmm_divide_var<<<grid_k,256>>>(d_new_var,d_Nk,K,feat);

        kernel_gmm_pi<<<grid_k,256>>>(d_pi,d_Nk,1.f/n,K);

        CUDA_CHECK(cudaMemcpy(d_mu, d_new_mu, K*feat*sizeof(float),cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_var,d_new_var,K*feat*sizeof(float),cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms=Ms(Clock::now()-t0).count();

    cudaFree(dX); cudaFree(d_mu); cudaFree(d_var); cudaFree(d_pi);
    cudaFree(d_resp); cudaFree(d_ll); cudaFree(d_Nk);
    cudaFree(d_new_mu); cudaFree(d_new_var);
    return {ms,log_lik,n,K};
}

// kernel pca — build rbf kernel matrix on gpu, then power iteration
__global__ void kernel_build_Knm(
        const float* __restrict__ X,
        const float* __restrict__ Lm,
        float* __restrict__ Knm,
        float inv_2sig2, int n, int m, int feat) {
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n||j>=m) return;
    float d=0.f;
    for(int f=0;f<feat;++f){
        float dd=X[i*feat+f]-Lm[j*feat+f];
        d+=dd*dd;
    }
    Knm[i*m+j]=expf(-d*inv_2sig2);
}

// matmul helpers
__global__ void kernel_matvec(const float* __restrict__ A, const float* __restrict__ v,
                               float* __restrict__ out, int rows, int cols) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows) return;
    float s=0.f;
    for(int j=0;j<cols;++j) s+=A[i*cols+j]*v[j];
    out[i]=s;
}

__global__ void kernel_matTvec(const float* __restrict__ A, const float* __restrict__ v,
                                float* __restrict__ out, int rows, int cols) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=cols) return;
    float s=0.f;
    for(int i=0;i<rows;++i) s+=A[i*cols+j]*v[i];
    out[j]=s;
}

__global__ void kernel_normalize(float* v, float* norm_out, int d){
    extern __shared__ float sdata[];
    int tid=threadIdx.x;
    sdata[tid]=0.f;
    for(int i=tid;i<d;i+=blockDim.x) sdata[tid]+=v[i]*v[i];
    __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){if(tid<s) sdata[tid]+=sdata[tid+s];__syncthreads();}
    if(tid==0) *norm_out=sqrtf(sdata[0]);
}

__global__ void kernel_vec_div_scalar(float* v, const float* s, int d){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<d) v[j]/=(*s+1e-10f);
}

struct KPCAResult { double time_ms,variance_explained; int n_samples; };

KPCAResult kpca_cuda(const std::vector<float>& hX, int n, int feat,
                      int m=64, float sigma2=1.f, int power_iter=20) {
    std::vector<int> lm_idx(n); std::iota(lm_idx.begin(),lm_idx.end(),0);
    std::shuffle(lm_idx.begin(),lm_idx.end(),G_RNG);
    lm_idx.resize(m);
    std::vector<float> h_Lm(m*feat);
    for(int i=0;i<m;++i) for(int j=0;j<feat;++j)
        h_Lm[i*feat+j]=hX[lm_idx[i]*feat+j];

    float *dX,*d_Lm,*d_Knm,*d_v,*d_tmp_m,*d_tmp_n,*d_norm;
    CUDA_CHECK(cudaMalloc(&dX,    n*feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Lm,  m*feat*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Knm, n*m*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v,   n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp_m, m*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp_n, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm,  sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX,  hX.data(),   n*feat*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Lm,h_Lm.data(),m*feat*sizeof(float),cudaMemcpyHostToDevice));

    std::vector<float> h_v(n, 1.f/sqrtf((float)n));
    CUDA_CHECK(cudaMemcpy(d_v,h_v.data(),n*sizeof(float),cudaMemcpyHostToDevice));

    dim3 blk2(16,16), grid2((m+15)/16,(n+15)/16);

    auto t0=Clock::now();

    kernel_build_Knm<<<grid2,blk2>>>(dX,d_Lm,d_Knm,1.f/(2.f*sigma2),n,m,feat);

    float eigenval=0.f;
    int grid_n2=(n+255)/256, grid_m2=(m+255)/256;
    for(int it=0;it<power_iter;++it){
        kernel_matTvec<<<grid_m2,256>>>(d_Knm,d_v,d_tmp_m,n,m);
        kernel_matvec<<<grid_n2,256>>>(d_Knm,d_tmp_m,d_tmp_n,n,m);
        kernel_normalize<<<1,256,256*sizeof(float)>>>(d_tmp_n,d_norm,n);
        float h_norm;
        CUDA_CHECK(cudaMemcpy(&h_norm,d_norm,sizeof(float),cudaMemcpyDeviceToHost));
        eigenval=h_norm;
        if(h_norm<1e-10f) break;
        kernel_vec_div_scalar<<<grid_n2,256>>>(d_tmp_n,d_norm,n);
        CUDA_CHECK(cudaMemcpy(d_v,d_tmp_n,n*sizeof(float),cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double ms=Ms(Clock::now()-t0).count();

    float total_var=gpu_sum(d_Knm, n*m);
    float var_expl=(total_var>0.f) ? (eigenval/(total_var/m))*100.f : 0.f;

    cudaFree(dX); cudaFree(d_Lm); cudaFree(d_Knm);
    cudaFree(d_v); cudaFree(d_tmp_m); cudaFree(d_tmp_n); cudaFree(d_norm);
    return {ms,(double)var_expl,n};
}

// mlp adamw — forward/backward, he init, full gpu training
// arch: feat -> 64(relu) -> 32(relu) -> K(softmax)
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

// weight grad: gW[o,j] = sum_i D_out[i,o] * A_in[i,j]
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

__global__ void kernel_bias_grad(const float* __restrict__ D_out, float* __restrict__ gb, int n, int out_dim) {
    int o=blockIdx.x*blockDim.x+threadIdx.x;
    if(o>=out_dim) return;
    float s=0.f;
    for(int i=0;i<n;++i) s+=D_out[i*out_dim+o];
    gb[o]=s;
}

struct MLPResult { double time_ms,final_loss,accuracy; int n_samples; };

MLPResult mlp_adamw_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
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

// random forest — trees built on cpu, prediction parallelized on gpu
struct FlatTree {
    std::vector<int>   feature;
    std::vector<float> threshold;
    std::vector<int>   left_child;
    std::vector<int>   right_child;
    std::vector<int>   label;
};

static void build_flat_tree(const std::vector<float>& X, const std::vector<int>& y,
                              int n, int feat, int K,
                              std::vector<int>& sample_idx,
                              int max_depth, int min_leaf,
                              FlatTree& tree, int depth) {
    int node_id=(int)tree.feature.size();
    tree.feature.push_back(-1);
    tree.threshold.push_back(0.f);
    tree.left_child.push_back(-1);
    tree.right_child.push_back(-1);
    tree.label.push_back(-1);

    bool is_leaf=false;
    if(depth>=max_depth || (int)sample_idx.size()<=min_leaf) is_leaf=true;
    if(!is_leaf){
        bool pure=true; int first=y[sample_idx[0]];
        for(int i:sample_idx) if(y[i]!=first){pure=false;break;}
        if(pure) is_leaf=true;
    }
    if(is_leaf){
        std::vector<int> cnt(K,0);
        for(int i:sample_idx) cnt[y[i]]++;
        tree.label[node_id]=(int)(std::max_element(cnt.begin(),cnt.end())-cnt.begin());
        return;
    }

    int n_try=std::max(1,(int)sqrtf((float)feat));
    std::vector<int> fs(feat); std::iota(fs.begin(),fs.end(),0);
    std::shuffle(fs.begin(),fs.end(),G_RNG); fs.resize(n_try);

    std::vector<int> cnt_p(K,0);
    for(int i:sample_idx) cnt_p[y[i]]++;
    int tot_p=(int)sample_idx.size();
    float gini_p=1.f;
    for(int k=0;k<K;++k){float p=(float)cnt_p[k]/tot_p; gini_p-=p*p;}

    float best_gain=-1e30f; int best_f=-1; float best_thr=0.f;
    std::vector<int> best_l,best_r;

    for(int f:fs){
        std::vector<std::pair<float,int>> vals;
        for(int i:sample_idx) vals.push_back({X[i*feat+f],i});
        std::sort(vals.begin(),vals.end());
        int sz=(int)vals.size();
        std::vector<int> l_idx,r_idx(sample_idx);
        for(int vi=0;vi<sz-1;++vi){
            l_idx.push_back(vals[vi].second);
            r_idx.erase(std::find(r_idx.begin(),r_idx.end(),vals[vi].second));
            if(vals[vi].first==vals[vi+1].first) continue;
            std::vector<int> lc(K,0); for(int i:l_idx) lc[y[i]]++;
            int nl=(int)l_idx.size(); float gl=1.f;
            for(int k=0;k<K;++k){float p=(float)lc[k]/nl; gl-=p*p;}
            std::vector<int> rc(K,0); for(int i:r_idx) rc[y[i]]++;
            int nr=(int)r_idx.size(); float gr=1.f;
            for(int k=0;k<K;++k){float p=(float)rc[k]/nr; gr-=p*p;}

            float thr=(vals[vi].first+vals[vi+1].first)*0.5f;
            float gain=gini_p-((float)nl/tot_p*gl+(float)nr/tot_p*gr);
            if(gain>best_gain){best_gain=gain;best_f=f;best_thr=thr;best_l=l_idx;best_r=r_idx;}
        }
    }
    if(best_f==-1||best_l.empty()||best_r.empty()){
        std::vector<int> cnt(K,0); for(int i:sample_idx) cnt[y[i]]++;
        tree.label[node_id]=(int)(std::max_element(cnt.begin(),cnt.end())-cnt.begin());
        return;
    }
    tree.feature[node_id]=best_f; tree.threshold[node_id]=best_thr;
    tree.left_child[node_id]=(int)tree.feature.size();
    build_flat_tree(X,y,n,feat,K,best_l,max_depth,min_leaf,tree,depth+1);
    tree.right_child[node_id]=(int)tree.feature.size();
    build_flat_tree(X,y,n,feat,K,best_r,max_depth,min_leaf,tree,depth+1);
}

__global__ void kernel_rf_predict(
        const float* __restrict__ X,
        const int*   __restrict__ feat_arr,
        const float* __restrict__ thr_arr,
        const int*   __restrict__ left_arr,
        const int*   __restrict__ right_arr,
        const int*   __restrict__ lbl_arr,
        int* __restrict__ votes,
        int n, int feat, int K) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int node=0;
    while(feat_arr[node]!=-1){
        if(X[i*feat+feat_arr[node]]<=thr_arr[node]) node=left_arr[node];
        else                                          node=right_arr[node];
    }
    atomicAdd(&votes[i*K+lbl_arr[node]],1);
}

struct RFResult { double time_ms,accuracy; int n_samples; };

RFResult rf_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
                  int n, int feat, int K,
                  int n_trees=20, int max_depth=8, int min_leaf=4) {
    std::vector<FlatTree> forest(n_trees);
    auto t_build=Clock::now();

    for(int t=0;t<n_trees;++t){
        std::vector<int> bag(n);
        for(int i=0;i<n;++i) bag[i]=randi(0,n);
        build_flat_tree(hX,hy,n,feat,K,bag,max_depth,min_leaf,forest[t],0);
    }
    double build_ms=Ms(Clock::now()-t_build).count();

    float* dX;
    CUDA_CHECK(cudaMalloc(&dX,n*feat*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX,hX.data(),n*feat*sizeof(float),cudaMemcpyHostToDevice));

    int* d_votes;
    CUDA_CHECK(cudaMalloc(&d_votes,n*K*sizeof(int)));
    CUDA_CHECK(cudaMemset(d_votes,0,n*K*sizeof(int)));

    auto t0=Clock::now();

    for(int t=0;t<n_trees;++t){
        const FlatTree& tree=forest[t];
        int sz=(int)tree.feature.size();

        int *d_feat,*d_left,*d_right,*d_lbl;
        float *d_thr;
        CUDA_CHECK(cudaMalloc(&d_feat,  sz*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_thr,   sz*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_left,  sz*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_right, sz*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_lbl,   sz*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_feat, tree.feature.data(),    sz*sizeof(int),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_thr,  tree.threshold.data(),  sz*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_left, tree.left_child.data(), sz*sizeof(int),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_right,tree.right_child.data(),sz*sizeof(int),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lbl,  tree.label.data(),      sz*sizeof(int),  cudaMemcpyHostToDevice));

        kernel_rf_predict<<<(n+255)/256,256>>>(dX,d_feat,d_thr,d_left,d_right,d_lbl,d_votes,n,feat,K);

        cudaFree(d_feat); cudaFree(d_thr); cudaFree(d_left);
        cudaFree(d_right); cudaFree(d_lbl);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double predict_ms=Ms(Clock::now()-t0).count();

    std::vector<int> h_votes(n*K);
    CUDA_CHECK(cudaMemcpy(h_votes.data(),d_votes,n*K*sizeof(int),cudaMemcpyDeviceToHost));
    int correct=0;
    for(int i=0;i<n;++i){
        int pred=0;
        for(int k=1;k<K;++k) if(h_votes[i*K+k]>h_votes[i*K+pred]) pred=k;
        if(pred==hy[i]) correct++;
    }
    cudaFree(dX); cudaFree(d_votes);
    return {build_ms+predict_ms, 100.0*correct/n, n};
}

static std::ofstream G_CSV;
static void write_row(const std::string& algo, int n, int feat,
                      double time_ms, const std::string& mname, double mval) {
    G_CSV << algo << "," << n << "," << feat << "," << time_ms
          << "," << mname << "," << mval << ",CUDA\n";
}

int main(int argc, char* argv[]) {
    int dev_count=0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if(dev_count==0){ std::cerr<<"[CUDA] No GPU found!\n"; return 1; }
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout<<"[CUDA] Device: "<<prop.name<<"\n";

    std::string out=(argc>1)?argv[1]:"results/cuda_results.csv";
    G_CSV.open(out);
    if(!G_CSV){ std::cerr<<"Cannot open "<<out<<"\n"; return 1; }
    G_CSV<<"algorithm,n_samples,n_features,time_ms,metric_name,metric_value,device\n";
    G_CSV<<std::fixed;

    std::vector<int> sizes={512,1024,2048,4096,8192,16384};
    const int FEAT=32, EPOCHS=60, BATCH=64, K_CLS=4;

    for(int n:sizes){
        std::cout<<"[CUDA] n="<<n<<"\n";

        std::vector<float> X_cls; std::vector<int> y_cls;
        gen_multiclass(X_cls,y_cls,n,FEAT,K_CLS);
        std::vector<int> y_bin(n); for(int i=0;i<n;++i) y_bin[i]=(y_cls[i]%2);

        std::cout<<"  AdamW ... "<<std::flush;
        { auto r=adamw_cuda(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("AdamW",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("AdamW",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms, acc="<<r.accuracy<<"%\n"; }

        std::cout<<"  Nadam ... "<<std::flush;
        { auto r=nadam_cuda(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("Nadam",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("Nadam",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  RMSProp ... "<<std::flush;
        { auto r=rmsprop_cuda(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("RMSProp",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("RMSProp",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  SGD_Nesterov ... "<<std::flush;
        { auto r=sgd_nesterov_cuda(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("SGD_Nesterov",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("SGD_Nesterov",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  SGDR ... "<<std::flush;
        { auto r=sgdr_cuda(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("SGDR",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("SGDR",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  L-BFGS ... "<<std::flush;
        { auto r=lbfgs_cuda(X_cls,y_bin,n,FEAT,50,10);
          write_row("LBFGS",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("LBFGS",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  GMM-EM ... "<<std::flush;
        { auto r=gmm_em_cuda(X_cls,n,FEAT,K_CLS,80);
          write_row("GMM_EM",n,FEAT,r.time_ms,"LogLikelihood",r.log_likelihood);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  Kernel PCA ... "<<std::flush;
        { auto r=kpca_cuda(X_cls,n,FEAT,std::min(n,128),1.f,30);
          write_row("KernelPCA",n,FEAT,r.time_ms,"VarExplained",r.variance_explained);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  MLP-AdamW ... "<<std::flush;
        { auto r=mlp_adamw_cuda(X_cls,y_cls,n,FEAT,K_CLS,EPOCHS,BATCH);
          write_row("MLP_AdamW",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("MLP_AdamW",n,FEAT,r.time_ms,"CE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  RandomForest ... "<<std::flush;
        { int n_rf=std::min(n,4096);
          std::vector<float> Xrf(X_cls.begin(),X_cls.begin()+n_rf*FEAT);
          std::vector<int>   yrf(y_cls.begin(),y_cls.begin()+n_rf);
          auto r=rf_cuda(Xrf,yrf,n_rf,FEAT,K_CLS,20,8,4);
          write_row("RandomForest",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          std::cout<<r.time_ms<<" ms\n"; }
    }

    G_CSV.close();
    std::cout<<"\n[CUDA] Results written to "<<out<<"\n";
    return 0;
}
