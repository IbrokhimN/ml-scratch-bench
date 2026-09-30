// Kernel PCA with a Nystrom approximation.
// An RBF kernel between all samples and a few random landmarks is centered
// and the top eigenvalue is found with power iteration.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "../common/rng.hpp"
#include "../common/timer.hpp"

// Result of a kernel PCA run.
struct KPCAResult { double time_ms, variance_explained; int n_samples; };

// Timing starts after the kernel matrices are built and covers centering and power iteration.
inline KPCAResult kernel_pca(const std::vector<float>& X, int n, int feat,
                             int m_landmarks=64, float sigma2=1.0f, int power_iter=20) {
    std::vector<int> lm(n); std::iota(lm.begin(),lm.end(),0);
    std::shuffle(lm.begin(),lm.end(),G_RNG);
    lm.resize(m_landmarks);

    auto rbf=[&](const float* a, const float* b)->float{
        float d=0.f;
        for(int j=0;j<feat;++j){float dd=a[j]-b[j]; d+=dd*dd;}
        return std::exp(-d/(2.f*sigma2));
    };

    // build Kmm
    std::vector<float> Kmm(m_landmarks*m_landmarks);
    for(int i=0;i<m_landmarks;++i)
        for(int j=0;j<m_landmarks;++j)
            Kmm[i*m_landmarks+j]=rbf(&X[lm[i]*feat],&X[lm[j]*feat]);

    // build Knm
    std::vector<float> Knm(n*m_landmarks);
    for(int i=0;i<n;++i)
        for(int j=0;j<m_landmarks;++j)
            Knm[i*m_landmarks+j]=rbf(&X[i*feat],&X[lm[j]*feat]);

    auto t0=Clock::now();

    std::vector<float> row_mean_Knm(n,0.f), col_mean_Knm(m_landmarks,0.f);
    float global_mean_Kmm=0.f;
    for(int i=0;i<n;++i)
        for(int j=0;j<m_landmarks;++j) row_mean_Knm[i]+=Knm[i*m_landmarks+j];
    for(int i=0;i<n;++i) row_mean_Knm[i]/=m_landmarks;
    for(int j=0;j<m_landmarks;++j)
        for(int i=0;i<n;++i) col_mean_Knm[j]+=Knm[i*m_landmarks+j];
    for(int j=0;j<m_landmarks;++j) col_mean_Knm[j]/=n;
    for(int v:lm)
        for(int j=0;j<m_landmarks;++j) global_mean_Kmm+=Kmm[0*m_landmarks+j];
    global_mean_Kmm/=(m_landmarks*m_landmarks);

    std::vector<float> Knm_c(n*m_landmarks);
    for(int i=0;i<n;++i)
        for(int j=0;j<m_landmarks;++j)
            Knm_c[i*m_landmarks+j] = Knm[i*m_landmarks+j]
                                    - row_mean_Knm[i]
                                    - col_mean_Knm[j]
                                    + global_mean_Kmm;

    // power iteration
    std::vector<float> v(n, 1.f/std::sqrt((float)n));
    float eigenval=0.f;
    for(int it=0;it<power_iter;++it){
        std::vector<float> Ktv(m_landmarks,0.f);
        for(int i=0;i<n;++i)
            for(int j=0;j<m_landmarks;++j) Ktv[j]+=Knm_c[i*m_landmarks+j]*v[i];
        std::vector<float> Kw(n,0.f);
        for(int i=0;i<n;++i)
            for(int j=0;j<m_landmarks;++j) Kw[i]+=Knm_c[i*m_landmarks+j]*Ktv[j];
        eigenval=0.f; for(float x:Kw) eigenval+=x*x; eigenval=std::sqrt(eigenval);
        if(eigenval<1e-10f) break;
        for(int i=0;i<n;++i) v[i]=Kw[i]/eigenval;
    }

    float trace=0.f;
    for(int i=0;i<n;++i) for(int j=0;j<m_landmarks;++j)
        trace+=Knm_c[i*m_landmarks+j]*Knm_c[i*m_landmarks+j];
    float var_expl = (trace>0.f) ? (eigenval/(trace/m_landmarks))*100.f : 0.f;

    double ms=Ms(Clock::now()-t0).count();
    return {ms, (double)var_expl, n};
}
