// SGD with cosine annealing and warm restarts for binary logistic regression.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "../common/math_ops.hpp"
#include "../common/rng.hpp"
#include "../common/timer.hpp"
#include "result.hpp"

// The first cycle lasts T0 epochs and every later cycle is T_mult times longer.
inline OptResult sgdr_logistic(const std::vector<float>& X, const std::vector<int>& y,
                                  int n, int feat, int epochs, int batch_size,
                                  float lr_max=0.05f, float lr_min=1e-5f,
                                  int T0=10, int T_mult=2) {
    std::vector<float> w(feat,0.f);
    float b=0.f;
    double loss=0.0;
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);

    int T_cur=0, T_i=T0;
    auto t0=Clock::now();

    for(int ep=0;ep<epochs;++ep){
        float lr = lr_min + 0.5f*(lr_max-lr_min)*(1.f+std::cos((float)M_PI*T_cur/T_i));
        T_cur++;
        if(T_cur>=T_i){ T_cur=0; T_i*=T_mult; }

        std::shuffle(idx.begin(),idx.end(),G_RNG);
        loss=0.0;
        for(int start=0;start<n;start+=batch_size){
            int end=std::min(start+batch_size,n), bs=end-start;
            std::vector<float> gw(feat,0.f); float gb=0.f,bl=0.f;
            for(int ii=start;ii<end;++ii){
                int i=idx[ii];
                float z=b; for(int j=0;j<feat;++j) z+=w[j]*X[i*feat+j];
                float p=sigmoid(z),yi=(float)y[i];
                bl+=-(yi*std::log(p+1e-9f)+(1.f-yi)*std::log(1.f-p+1e-9f));
                float dz=p-yi;
                for(int j=0;j<feat;++j) gw[j]+=dz*X[i*feat+j];
                gb+=dz;
            }
            for(int j=0;j<feat;++j){gw[j]/=bs; w[j]-=lr*gw[j];}
            b-=lr*gb/bs;
            loss+=bl/bs;
        }
        loss/=(double)(n/batch_size+1);
    }
    double ms=Ms(Clock::now()-t0).count();
    int correct=0;
    for(int i=0;i<n;++i){
        float z=b; for(int j=0;j<feat;++j) z+=w[j]*X[i*feat+j];
        if((sigmoid(z)>=0.5f?1:0)==y[i]) correct++;
    }
    return {ms,loss,100.0*correct/n,n,"SGDR"};
}
