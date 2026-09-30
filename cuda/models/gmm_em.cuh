// Gaussian mixture model with diagonal covariance fitted by EM on the GPU.
// The E step and every M step reduction run as kernels.
#pragma once

#include <algorithm>
#include <vector>

#include "../common/cuda_utils.cuh"
#include "../common/rng.hpp"
#include "../common/timer.hpp"
#include "../kernels/reduce.cuh"

// One thread per sample. Computes log responsibilities for every component and normalizes them.
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

// One thread per sample. Adds responsibilities into the per component counts.
__global__ void kernel_gmm_m_nk(const float* __restrict__ resp, float* __restrict__ Nk, int n, int K) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    for(int k=0;k<K;++k) atomicAdd(&Nk[k], resp[i*K+k]);
}

// One thread per sample. Adds responsibility weighted points into the new means.
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

// One thread per component. Turns the weighted sums into means.
__global__ void kernel_gmm_divide_mu(float* __restrict__ mu, const float* __restrict__ Nk, int K, int feat) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=K) return;
    float nk=fmaxf(Nk[k],1e-6f);
    for(int j=0;j<feat;++j) mu[k*feat+j]/=nk;
}

// One thread per sample. Adds weighted squared deviations into the new variances.
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

// One thread per component. Turns the sums into variances and adds a small floor.
__global__ void kernel_gmm_divide_var(float* __restrict__ var, const float* __restrict__ Nk, int K, int feat) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=K) return;
    float nk=fmaxf(Nk[k],1e-6f);
    for(int j=0;j<feat;++j) var[k*feat+j]=var[k*feat+j]/nk+1e-4f;
}

// One thread per component. Updates the mixing weights from the counts.
__global__ void kernel_gmm_pi(float* pi, const float* Nk, float n_inv, int K){
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k<K) pi[k]=fmaxf(Nk[k],1e-6f)*n_inv;
}

// Result of a GMM run. The likelihood is averaged over samples.
struct GMMResult { double time_ms,log_likelihood; int n_samples,K; };

// Picks initial means with kmeans++ on the host and then runs EM on the device.
// Timing covers the EM loop only.
inline GMMResult gmm_em_cuda(const std::vector<float>& hX, int n, int feat, int K, int max_iter=100) {
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
        // E step
        kernel_gmm_e_step<<<grid_n,256>>>(dX,d_mu,d_var,d_pi,d_resp,d_ll,n,feat,K);
        log_lik=(double)gpu_sum(d_ll,n)/n;

        // M step
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
