// RMSProp with momentum for binary logistic regression.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "../common/math_ops.hpp"
#include "../common/rng.hpp"
#include "../common/timer.hpp"
#include "result.hpp"

// Same interface as adamw_logistic.
inline OptResult rmsprop_logistic(const std::vector<float>& X, const std::vector<int>& y,
                                     int n, int feat, int epochs, int batch_size,
                                     float lr=1e-3f, float rho=0.9f,
                                     float eps=1e-8f, float momentum=0.9f) {
    std::vector<float> w(feat,0.f), eg2(feat,0.f), delta_w(feat,0.f);
    float b=0.f, eg2_b=0.f, delta_b=0.f;
    double loss=0.0;
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);

    auto t0 = Clock::now();

    for(int ep=0;ep<epochs;++ep){
        std::shuffle(idx.begin(),idx.end(),G_RNG);
        loss=0.0;
        for(int start=0;start<n;start+=batch_size){
            int end=std::min(start+batch_size,n), bs=end-start;
            std::vector<float> gw(feat,0.f); float gb=0.f,bl=0.f;
            for(int ii=start;ii<end;++ii){
                int i=idx[ii];
                float z=b; for(int j=0;j<feat;++j) z+=w[j]*X[i*feat+j];
                float p=sigmoid(z), yi=(float)y[i];
                bl+=-(yi*std::log(p+1e-9f)+(1.f-yi)*std::log(1.f-p+1e-9f));
                float dz=p-yi;
                for(int j=0;j<feat;++j) gw[j]+=dz*X[i*feat+j];
                gb+=dz;
            }
            for(int j=0;j<feat;++j) gw[j]/=bs; gb/=bs;
            loss+=bl/bs;
            for(int j=0;j<feat;++j){
                eg2[j] = rho*eg2[j] + (1.f-rho)*gw[j]*gw[j];
                delta_w[j] = momentum*delta_w[j] - lr * gw[j] / (std::sqrt(eg2[j]) + eps);
                w[j] += delta_w[j];
            }
            eg2_b   = rho*eg2_b   + (1.f-rho)*gb*gb;
            delta_b = momentum*delta_b - lr*gb/(std::sqrt(eg2_b)+eps);
            b      += delta_b;
        }
        loss/=(double)(n/batch_size+1);
    }
    double ms=Ms(Clock::now()-t0).count();
    int correct=0;
    for(int i=0;i<n;++i){
        float z=b; for(int j=0;j<feat;++j) z+=w[j]*X[i*feat+j];
        if((sigmoid(z)>=0.5f?1:0)==y[i]) correct++;
    }
    return {ms,loss,100.0*correct/n,n,"RMSProp"};
}
