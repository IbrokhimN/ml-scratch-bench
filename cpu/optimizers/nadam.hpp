// Nadam for binary logistic regression.
// Adam with a Nesterov style lookahead applied to the first moment.
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
inline OptResult nadam_logistic(const std::vector<float>& X, const std::vector<int>& y,
                                   int n, int feat, int epochs, int batch_size,
                                   float lr=1e-3f, float beta1=0.9f,
                                   float beta2=0.999f, float eps=1e-8f) {
    std::vector<float> w(feat, 0.f), m_w(feat, 0.f), v_w(feat, 0.f);
    float b=0.f, m_b=0.f, v_b=0.f;
    int t = 0;
    double loss = 0.0;
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);

    auto t0 = Clock::now();

    for (int ep = 0; ep < epochs; ++ep) {
        std::shuffle(idx.begin(), idx.end(), G_RNG);
        loss = 0.0;

        for (int start = 0; start < n; start += batch_size) {
            int end = std::min(start + batch_size, n);
            int bs  = end - start;
            ++t;

            std::vector<float> gw(feat, 0.f);
            float gb = 0.f, bl = 0.f;

            for (int ii = start; ii < end; ++ii) {
                int i = idx[ii];
                float z = b;
                for (int j = 0; j < feat; ++j) z += w[j] * X[i*feat+j];
                float p  = sigmoid(z);
                float yi = (float)y[i];
                bl += -(yi*std::log(p+1e-9f) + (1.f-yi)*std::log(1.f-p+1e-9f));
                float dz = p - yi;
                for (int j = 0; j < feat; ++j) gw[j] += dz * X[i*feat+j];
                gb += dz;
            }
            for (int j=0;j<feat;++j) gw[j]/=bs; gb/=bs;
            loss += bl / bs;

            float bc1 = 1.f - std::pow(beta1,(float)t);
            float bc2 = 1.f - std::pow(beta2,(float)t);
            float bc1_next = 1.f - std::pow(beta1,(float)(t+1));

            for (int j = 0; j < feat; ++j) {
                m_w[j] = beta1*m_w[j] + (1.f-beta1)*gw[j];
                v_w[j] = beta2*v_w[j] + (1.f-beta2)*gw[j]*gw[j];
                float v_hat = v_w[j] / bc2;
                float nadam_term = (beta1 * m_w[j]/bc1_next) + ((1.f-beta1) * gw[j]/bc1);
                w[j] -= lr * nadam_term / (std::sqrt(v_hat) + eps);
            }
            m_b = beta1*m_b + (1.f-beta1)*gb;
            v_b = beta2*v_b + (1.f-beta2)*gb*gb;
            float v_hat_b = v_b / bc2;
            float nadam_b = (beta1*m_b/(1.f-std::pow(beta1,(float)(t+1)))) + ((1.f-beta1)*gb/bc1);
            b -= lr * nadam_b / (std::sqrt(v_hat_b) + eps);
        }
        loss /= (double)(n/batch_size+1);
    }
    double ms = Ms(Clock::now()-t0).count();
    int correct=0;
    for(int i=0;i<n;++i){
        float z=b; for(int j=0;j<feat;++j) z+=w[j]*X[i*feat+j];
        if((sigmoid(z)>=0.5f?1:0)==y[i]) correct++;
    }
    return {ms, loss, 100.0*correct/n, n, "Nadam"};
}
