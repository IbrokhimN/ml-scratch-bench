// LBFGS for binary logistic regression.
// Uses the two loop recursion over a short history and an Armijo backtracking line search.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "../common/math_ops.hpp"
#include "../common/timer.hpp"

// Result of an LBFGS run.
struct LBFGSResult {
    double time_ms, final_loss, accuracy;
    int    n_samples;
};

// Dot product of two vectors.
static float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float s=0.f;
    for(size_t i=0;i<a.size();++i) s+=a[i]*b[i];
    return s;
}

// Full batch BCE loss and gradient. The last gradient entry belongs to the bias.
static double bce_loss_grad(const std::vector<float>& X, const std::vector<int>& y,
                             const std::vector<float>& theta,
                             int n, int feat, std::vector<float>& grad) {
    grad.assign(feat+1, 0.f);
    double loss=0.0;
    for(int i=0;i<n;++i){
        float z=theta[feat];
        for(int j=0;j<feat;++j) z+=theta[j]*X[i*feat+j];
        float p=sigmoid(z), yi=(float)y[i];
        loss+=-(yi*std::log(p+1e-9f)+(1.f-yi)*std::log(1.f-p+1e-9f));
        float dz=p-yi;
        for(int j=0;j<feat;++j) grad[j]+=dz*X[i*feat+j];
        grad[feat]+=dz;
    }
    for(auto& g:grad) g/=n;
    return loss/n;
}

// Runs at most max_iter iterations and stops early once the gradient is tiny.
// M is the number of history pairs kept.
inline LBFGSResult lbfgs_logistic(const std::vector<float>& X, const std::vector<int>& y,
                                    int n, int feat, int max_iter=100, int M=10) {
    int d=feat+1;
    std::vector<float> theta(d, 0.f);
    std::vector<float> grad(d, 0.f);

    std::vector<std::vector<float>> s_hist(M, std::vector<float>(d,0.f));
    std::vector<std::vector<float>> y_hist(M, std::vector<float>(d,0.f));
    std::vector<float> rho_hist(M, 0.f);
    int history_len=0, head=0;

    auto t0=Clock::now();

    double loss=bce_loss_grad(X,y,theta,n,feat,grad);

    for(int iter=0;iter<max_iter;++iter){
        // two loop recursion
        std::vector<float> q=grad;
        std::vector<float> alpha_arr(M,0.f);

        for(int i=history_len-1;i>=0;--i){
            int idx=(head-1-i+2*M)%M;
            float ai=rho_hist[idx]*dot(s_hist[idx],q);
            alpha_arr[i]=ai;
            for(int k=0;k<d;++k) q[k]-=ai*y_hist[idx][k];
        }

        float gamma=1.f;
        if(history_len>0){
            int last=(head-1+M)%M;
            float sy=dot(s_hist[last],y_hist[last]);
            float yy=dot(y_hist[last],y_hist[last]);
            if(yy>1e-10f) gamma=sy/yy;
        }
        std::vector<float> r(d);
        for(int k=0;k<d;++k) r[k]=gamma*q[k];

        for(int i=0;i<history_len;++i){
            int idx=(head-history_len+i+2*M)%M;
            float beta=rho_hist[idx]*dot(y_hist[idx],r);
            for(int k=0;k<d;++k) r[k]+=s_hist[idx][k]*(alpha_arr[i]-beta);
        }

        std::vector<float> p(d);
        for(int k=0;k<d;++k) p[k]=-r[k];

        // armijo line search
        float step=1.0f;
        float c1=1e-4f;
        float pg=dot(p,grad);
        std::vector<float> theta_new(d), grad_new(d);
        double loss_new;
        int ls_iter=0;
        do {
            for(int k=0;k<d;++k) theta_new[k]=theta[k]+step*p[k];
            loss_new=bce_loss_grad(X,y,theta_new,n,feat,grad_new);
            if(loss_new <= loss + c1*step*pg) break;
            step*=0.5f;
        } while(++ls_iter<30 && step>1e-10f);

        auto& s_new=s_hist[head];
        auto& y_new=y_hist[head];
        for(int k=0;k<d;++k){ s_new[k]=theta_new[k]-theta[k]; y_new[k]=grad_new[k]-grad[k]; }
        float sy=dot(s_new,y_new);
        rho_hist[head]=(sy>1e-10f) ? 1.f/sy : 0.f;
        head=(head+1)%M;
        if(history_len<M) history_len++;

        theta=theta_new;
        grad=grad_new;
        loss=loss_new;

        float gnorm=0.f;
        for(float g:grad) gnorm=std::max(gnorm,std::abs(g));
        if(gnorm<1e-5f) break;
    }

    double ms=Ms(Clock::now()-t0).count();
    int correct=0;
    for(int i=0;i<n;++i){
        float z=theta[feat]; for(int j=0;j<feat;++j) z+=theta[j]*X[i*feat+j];
        if((sigmoid(z)>=0.5f?1:0)==y[i]) correct++;
    }
    return {ms, loss, 100.0*correct/n, n};
}
