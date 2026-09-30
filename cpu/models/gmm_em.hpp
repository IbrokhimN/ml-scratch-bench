// Gaussian mixture model with diagonal covariance fitted by EM.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "../common/rng.hpp"
#include "../common/timer.hpp"

// Result of a GMM run. The likelihood is averaged over samples.
struct GMMResult { double time_ms, log_likelihood; int n_samples, K; };

// Log density of a diagonal gaussian at point x.
static float log_gaussian(const float* x, const float* mu, const float* var, int d) {
    float lp = 0.f;
    const float LOG_2PI = 1.8378770664093455f;
    for(int j=0;j<d;++j){
        float diff = x[j]-mu[j];
        lp += -0.5f*(LOG_2PI + std::log(var[j]+1e-6f) + diff*diff/(var[j]+1e-6f));
    }
    return lp;
}

// Picks initial means with kmeans++ and then runs a fixed number of EM iterations.
// Timing covers the EM loop only.
inline GMMResult gmm_em(const std::vector<float>& X, int n, int feat, int K, int max_iter=100) {
    std::vector<float> mu(K*feat), var(K*feat, 1.f), pi(K, 1.f/K);

    // kmeans++ init
    std::vector<int> chosen;
    chosen.push_back(randi(0,n));
    for(int c=1;c<K;++c){
        std::vector<float> dist2(n, 1e30f);
        for(int i=0;i<n;++i)
            for(int prev:chosen){
                float d=0.f;
                for(int j=0;j<feat;++j){float dd=X[i*feat+j]-X[prev*feat+j];d+=dd*dd;}
                dist2[i]=std::min(dist2[i],d);
            }
        float total=0.f; for(float v:dist2) total+=v;
        float r=randf()*total, cum=0.f;
        int pick=n-1;
        for(int i=0;i<n;++i){ cum+=dist2[i]; if(cum>=r){pick=i;break;} }
        chosen.push_back(pick);
    }
    for(int c=0;c<K;++c)
        for(int j=0;j<feat;++j) mu[c*feat+j]=X[chosen[c]*feat+j];

    std::vector<float> resp(n*K);
    auto t0=Clock::now();

    double log_lik=0.0;
    for(int iter=0;iter<max_iter;++iter){
        // E step
        log_lik=0.0;
        for(int i=0;i<n;++i){
            float mx=-1e30f;
            std::vector<float> log_p(K);
            for(int k=0;k<K;++k){
                log_p[k] = std::log(pi[k]+1e-9f) + log_gaussian(&X[i*feat],&mu[k*feat],&var[k*feat],feat);
                mx=std::max(mx,log_p[k]);
            }
            float sum_exp=0.f;
            for(int k=0;k<K;++k){ resp[i*K+k]=std::exp(log_p[k]-mx); sum_exp+=resp[i*K+k]; }
            log_lik+=std::log(sum_exp)+mx;
            for(int k=0;k<K;++k) resp[i*K+k]/=sum_exp;
        }
        log_lik/=n;

        // M step
        std::vector<float> Nk(K,0.f);
        std::vector<float> new_mu(K*feat,0.f), new_var(K*feat,0.f);
        for(int i=0;i<n;++i)
            for(int k=0;k<K;++k){
                float r=resp[i*K+k];
                Nk[k]+=r;
                for(int j=0;j<feat;++j) new_mu[k*feat+j]+=r*X[i*feat+j];
            }
        for(int k=0;k<K;++k){
            float nk=std::max(Nk[k],1e-6f);
            pi[k]=nk/n;
            for(int j=0;j<feat;++j) new_mu[k*feat+j]/=nk;
        }
        for(int i=0;i<n;++i)
            for(int k=0;k<K;++k){
                float r=resp[i*K+k];
                for(int j=0;j<feat;++j){
                    float d=X[i*feat+j]-new_mu[k*feat+j];
                    new_var[k*feat+j]+=r*d*d;
                }
            }
        for(int k=0;k<K;++k){
            float nk=std::max(Nk[k],1e-6f);
            for(int j=0;j<feat;++j) new_var[k*feat+j]=new_var[k*feat+j]/nk + 1e-4f;
        }
        mu=new_mu; var=new_var;
    }

    double ms=Ms(Clock::now()-t0).count();
    return {ms, log_lik, n, K};
}
