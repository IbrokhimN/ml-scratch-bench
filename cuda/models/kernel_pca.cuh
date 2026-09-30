// Kernel PCA with a Nystrom approximation on the GPU.
// The RBF matrix is built with a 2D grid and power iteration runs on the device.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "../common/cuda_utils.cuh"
#include "../common/rng.hpp"
#include "../common/timer.hpp"
#include "../kernels/reduce.cuh"

// 2D grid with one thread per sample and landmark pair. Fills the RBF kernel matrix.
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

// Matrix vector helpers for power iteration.
// out equals A times v.
__global__ void kernel_matvec(const float* __restrict__ A, const float* __restrict__ v,
                               float* __restrict__ out, int rows, int cols) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows) return;
    float s=0.f;
    for(int j=0;j<cols;++j) s+=A[i*cols+j]*v[j];
    out[i]=s;
}

// out equals A transpose times v.
__global__ void kernel_matTvec(const float* __restrict__ A, const float* __restrict__ v,
                                float* __restrict__ out, int rows, int cols) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=cols) return;
    float s=0.f;
    for(int i=0;i<rows;++i) s+=A[i*cols+j]*v[i];
    out[j]=s;
}

// L2 norm of v computed by a single block and written to norm_out.
__global__ void kernel_normalize(float* v, float* norm_out, int d){
    extern __shared__ float sdata[];
    int tid=threadIdx.x;
    sdata[tid]=0.f;
    for(int i=tid;i<d;i+=blockDim.x) sdata[tid]+=v[i]*v[i];
    __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){if(tid<s) sdata[tid]+=sdata[tid+s];__syncthreads();}
    if(tid==0) *norm_out=sqrtf(sdata[0]);
}

// Divides v by a scalar that lives on the device.
__global__ void kernel_vec_div_scalar(float* v, const float* s, int d){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<d) v[j]/=(*s+1e-10f);
}

// Result of a kernel PCA run.
struct KPCAResult { double time_ms,variance_explained; int n_samples; };

// Timing covers the kernel matrix build and power iteration.
// Unlike the CPU version the matrix is not centered.
inline KPCAResult kpca_cuda(const std::vector<float>& hX, int n, int feat,
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
