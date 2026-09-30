// LBFGS for binary logistic regression on the GPU.
// History vectors, dot products and vector updates all stay on the device.
// Only scalars go back to the host for the recursion and the line search.
#pragma once

#include <cmath>
#include <vector>

#include "../common/cuda_utils.cuh"
#include "../common/timer.hpp"
#include "../kernels/device_math.cuh"
#include "../kernels/reduce.cuh"

// One thread per sample. Computes the BCE loss and atomically accumulates the gradient.
// The last gradient entry belongs to the bias.
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

// Dot product computed by a single block.
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

// r plus alpha times v.
__global__ void kernel_axpy(float* r, const float* v, float alpha, int d){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<d) r[j]+=alpha*v[j];
}

// r minus alpha times v.
__global__ void kernel_axpy_neg(float* r, const float* v, float alpha, int d){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<d) r[j]-=alpha*v[j];
}

// out equals in times s.
__global__ void kernel_scale(float* out, const float* in, float s, int d){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<d) out[j]=in[j]*s;
}

// Result of an LBFGS run.
struct LBFGSResult { double time_ms,final_loss,accuracy; int n_samples; };

// Runs exactly max_iter iterations. M is the number of history pairs kept.
inline LBFGSResult lbfgs_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
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
        // two loop recursion
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
