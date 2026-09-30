// Fully connected network trained with AdamW.
// Layout is feat to 64 to 32 to K with ReLU hidden layers and a softmax output.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "../common/math_ops.hpp"
#include "../common/rng.hpp"
#include "../common/timer.hpp"

// Result of an MLP run.
struct MLPResult { double time_ms, final_loss, accuracy; int n_samples; };

// He normal initialization for a weight matrix with the given fan in.
static void he_init(std::vector<float>& W, int fan_in, int fan_out) {
    float std = std::sqrt(2.f / fan_in);
    W.resize(fan_out * fan_in);
    for(auto& v:W) v=randnf()*std;
}

// Gradients are accumulated sample by sample over a mini batch and one AdamW step is taken per batch.
inline MLPResult mlp_adamw(const std::vector<float>& X, const std::vector<int>& y,
                            int n, int feat, int K, int epochs, int batch_size,
                            float lr=1e-3f, float wd=1e-4f) {
    const int H1=64, H2=32;

    std::vector<float> W1,b1(H1,0.f),W2,b2(H2,0.f),W3,b3(K,0.f);
    he_init(W1,feat,H1); he_init(W2,H1,H2); he_init(W3,H2,K);

    auto zero_like=[](const std::vector<float>& v){ return std::vector<float>(v.size(),0.f); };
    std::vector<float> mW1=zero_like(W1),vW1=zero_like(W1);
    std::vector<float> mb1(H1,0.f),vb1(H1,0.f);
    std::vector<float> mW2=zero_like(W2),vW2=zero_like(W2);
    std::vector<float> mb2(H2,0.f),vb2(H2,0.f);
    std::vector<float> mW3=zero_like(W3),vW3=zero_like(W3);
    std::vector<float> mb3(K,0.f),vb3(K,0.f);

    const float beta1=0.9f,beta2=0.999f,eps=1e-8f;
    int t=0;

    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);
    double loss=0.0;

    std::vector<float> z1(H1),a1(H1),z2(H2),a2(H2),z3(K),a3(K);
    std::vector<float> d3(K),d2(H2),d1(H1);
    std::vector<float> gW1(H1*feat),gb1(H1),gW2(H2*H1),gb2(H2),gW3(K*H2),gb3(K);

    auto t0=Clock::now();

    for(int ep=0;ep<epochs;++ep){
        std::shuffle(idx.begin(),idx.end(),G_RNG);
        loss=0.0;
        for(int start=0;start<n;start+=batch_size){
            int end=std::min(start+batch_size,n), bs=end-start;
            ++t;
            std::fill(gW1.begin(),gW1.end(),0.f); std::fill(gb1.begin(),gb1.end(),0.f);
            std::fill(gW2.begin(),gW2.end(),0.f); std::fill(gb2.begin(),gb2.end(),0.f);
            std::fill(gW3.begin(),gW3.end(),0.f); std::fill(gb3.begin(),gb3.end(),0.f);
            float bl=0.f;

            for(int ii=start;ii<end;++ii){
                int i=idx[ii];
                const float* xi=&X[i*feat];

                // forward
                for(int h=0;h<H1;++h){
                    z1[h]=b1[h];
                    for(int j=0;j<feat;++j) z1[h]+=W1[h*feat+j]*xi[j];
                    a1[h]=relu(z1[h]);
                }
                for(int h=0;h<H2;++h){
                    z2[h]=b2[h];
                    for(int j=0;j<H1;++j) z2[h]+=W2[h*H1+j]*a1[j];
                    a2[h]=relu(z2[h]);
                }
                for(int k=0;k<K;++k){
                    z3[k]=b3[k];
                    for(int j=0;j<H2;++j) z3[k]+=W3[k*H2+j]*a2[j];
                    a3[k]=z3[k];
                }
                softmax_inplace(a3.data(),K);
                bl -= std::log(a3[y[i]]+1e-9f);

                // backward
                for(int k=0;k<K;++k) d3[k]=(a3[k]-(k==y[i]?1.f:0.f))/bs;
                for(int k=0;k<K;++k){
                    gb3[k]+=d3[k];
                    for(int j=0;j<H2;++j) gW3[k*H2+j]+=d3[k]*a2[j];
                }
                for(int h=0;h<H2;++h){
                    float s=0.f;
                    for(int k=0;k<K;++k) s+=W3[k*H2+h]*d3[k];
                    d2[h]=s*relu_d(z2[h]);
                }
                for(int h=0;h<H2;++h){
                    gb2[h]+=d2[h];
                    for(int j=0;j<H1;++j) gW2[h*H1+j]+=d2[h]*a1[j];
                }
                for(int h=0;h<H1;++h){
                    float s=0.f;
                    for(int k=0;k<H2;++k) s+=W2[k*H1+h]*d2[k];
                    d1[h]=s*relu_d(z1[h]);
                }
                for(int h=0;h<H1;++h){
                    gb1[h]+=d1[h];
                    for(int j=0;j<feat;++j) gW1[h*feat+j]+=d1[h]*xi[j];
                }
            }
            loss+=bl/bs;

            float bc1=1.f-std::pow(beta1,(float)t);
            float bc2=1.f-std::pow(beta2,(float)t);

            // adamw update
            auto adam_update=[&](std::vector<float>& w, std::vector<float>& mw,
                                  std::vector<float>& vw, const std::vector<float>& gw,
                                  bool apply_wd){
                for(size_t k=0;k<w.size();++k){
                    float g=gw[k];
                    mw[k]=beta1*mw[k]+(1.f-beta1)*g;
                    vw[k]=beta2*vw[k]+(1.f-beta2)*g*g;
                    float mh=mw[k]/bc1, vh=vw[k]/bc2;
                    float step_val=lr*mh/(std::sqrt(vh)+eps);
                    if(apply_wd) step_val+=lr*wd*w[k];
                    w[k]-=step_val;
                }
            };
            adam_update(W1,mW1,vW1,gW1,true);  adam_update(b1,mb1,vb1,gb1,false);
            adam_update(W2,mW2,vW2,gW2,true);  adam_update(b2,mb2,vb2,gb2,false);
            adam_update(W3,mW3,vW3,gW3,true);  adam_update(b3,mb3,vb3,gb3,false);
        }
        loss/=(double)(n/batch_size+1);
    }
    double ms=Ms(Clock::now()-t0).count();

    int correct=0;
    for(int i=0;i<n;++i){
        const float* xi=&X[i*feat];
        for(int h=0;h<H1;++h){
            z1[h]=b1[h];
            for(int j=0;j<feat;++j) z1[h]+=W1[h*feat+j]*xi[j];
            a1[h]=relu(z1[h]);
        }
        for(int h=0;h<H2;++h){
            z2[h]=b2[h];
            for(int j=0;j<H1;++j) z2[h]+=W2[h*H1+j]*a1[j];
            a2[h]=relu(z2[h]);
        }
        for(int k=0;k<K;++k){
            z3[k]=b3[k];
            for(int j=0;j<H2;++j) z3[k]+=W3[k*H2+j]*a2[j];
            a3[k]=z3[k];
        }
        int pred=(int)(std::max_element(a3.begin(),a3.end())-a3.begin());
        if(pred==y[i]) correct++;
    }
    return {ms, loss, 100.0*correct/n, n};
}
