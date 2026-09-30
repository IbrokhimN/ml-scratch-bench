// SGD with Nesterov momentum for binary logistic regression.
// The learning rate follows a cosine decay and never drops below one percent of its start value.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "../common/math_ops.hpp"
#include "../common/rng.hpp"
#include "../common/timer.hpp"
#include "result.hpp"

// Gradients are evaluated at the lookahead point, not at the current weights.
inline OptResult sgd_nesterov_logistic(const std::vector<float>& X, const std::vector<int>& y,
                                          int n, int feat, int epochs, int batch_size,
                                          float lr=0.01f, float mu=0.9f) {
    std::vector<float> w(feat,0.f), vel(feat,0.f);
    float b=0.f, vel_b=0.f;
    double loss=0.0;
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);

    auto t0 = Clock::now();

    for(int ep=0;ep<epochs;++ep){
        float lr_t = lr * 0.5f * (1.f + std::cos((float)M_PI * ep / epochs));
        lr_t = std::max(lr_t, lr * 0.01f);

        std::shuffle(idx.begin(),idx.end(),G_RNG);
        loss=0.0;
        for(int start=0;start<n;start+=batch_size){
            int end=std::min(start+batch_size,n), bs=end-start;
            std::vector<float> w_la(feat);
            for(int j=0;j<feat;++j) w_la[j]=w[j]+mu*vel[j];
            float b_la=b+mu*vel_b;

            std::vector<float> gw(feat,0.f); float gb=0.f,bl=0.f;
            for(int ii=start;ii<end;++ii){
                int i=idx[ii];
                float z=b_la;
                for(int j=0;j<feat;++j) z+=w_la[j]*X[i*feat+j];
                float p=sigmoid(z),yi=(float)y[i];
                bl+=-(yi*std::log(p+1e-9f)+(1.f-yi)*std::log(1.f-p+1e-9f));
                float dz=p-yi;
                for(int j=0;j<feat;++j) gw[j]+=dz*X[i*feat+j];
                gb+=dz;
            }
            for(int j=0;j<feat;++j) gw[j]/=bs; gb/=bs;
            loss+=bl/bs;

            for(int j=0;j<feat;++j){
                vel[j] = mu*vel[j] - lr_t*gw[j];
                w[j]  += vel[j];
            }
            vel_b = mu*vel_b - lr_t*gb;
            b    += vel_b;
        }
        loss/=(double)(n/batch_size+1);
    }
    double ms=Ms(Clock::now()-t0).count();
    int correct=0;
    for(int i=0;i<n;++i){
        float z=b; for(int j=0;j<feat;++j) z+=w[j]*X[i*feat+j];
        if((sigmoid(z)>=0.5f?1:0)==y[i]) correct++;
    }
    return {ms,loss,100.0*correct/n,n,"SGD_Nesterov"};
}
