// AdamW for binary logistic regression.
// Mini batch gradients, bias corrected moments and decoupled weight decay.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "../common/math_ops.hpp"
#include "../common/rng.hpp"
#include "../common/timer.hpp"
#include "result.hpp"

// Trains on X with 0 or 1 labels y. Returns the training time, last epoch loss and accuracy.
inline OptResult adamw_logistic(const std::vector<float>& X, const std::vector<int>& y,
                                   int n, int feat, int epochs, int batch_size,
                                   float lr=1e-3f, float beta1=0.9f,
                                   float beta2=0.999f, float eps=1e-8f, float wd=1e-2f) {
    std::vector<float> w(feat, 0.f), m_w(feat, 0.f), v_w(feat, 0.f);
    float b = 0.f, m_b = 0.f, v_b = 0.f;
    int t = 0;
    double loss = 0.0;
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    auto t0 = Clock::now();

    for (int ep = 0; ep < epochs; ++ep) {
        std::shuffle(idx.begin(), idx.end(), G_RNG);
        loss = 0.0;

        for (int start = 0; start < n; start += batch_size) {
            int end = std::min(start + batch_size, n);
            int bs  = end - start;
            ++t;

            std::vector<float> gw(feat, 0.f);
            float gb = 0.f;
            float batch_loss = 0.f;

            for (int ii = start; ii < end; ++ii) {
                int i = idx[ii];
                float z = b;
                for (int j = 0; j < feat; ++j) z += w[j] * X[i*feat+j];
                float p   = sigmoid(z);
                float yi  = (float)y[i];
                batch_loss += -(yi * std::log(p + 1e-9f) + (1.f-yi) * std::log(1.f-p + 1e-9f));
                float dz = p - yi;
                for (int j = 0; j < feat; ++j) gw[j] += dz * X[i*feat+j];
                gb += dz;
            }

            for (int j = 0; j < feat; ++j) gw[j] /= bs;
            gb /= bs;
            loss += batch_loss / bs;

            float bc1 = 1.f - std::pow(beta1, (float)t);
            float bc2 = 1.f - std::pow(beta2, (float)t);

            for (int j = 0; j < feat; ++j) {
                m_w[j] = beta1 * m_w[j] + (1.f-beta1) * gw[j];
                v_w[j] = beta2 * v_w[j] + (1.f-beta2) * gw[j]*gw[j];
                float m_hat = m_w[j] / bc1;
                float v_hat = v_w[j] / bc2;
                w[j] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + wd * w[j]);
            }
            m_b = beta1 * m_b + (1.f-beta1) * gb;
            v_b = beta2 * v_b + (1.f-beta2) * gb*gb;
            float m_hat_b = m_b / bc1;
            float v_hat_b = v_b / bc2;
            b -= lr * m_hat_b / (std::sqrt(v_hat_b) + eps);
        }
        loss /= (double)(n / batch_size + 1);
    }

    double ms = Ms(Clock::now() - t0).count();

    int correct = 0;
    for (int i = 0; i < n; ++i) {
        float z = b;
        for (int j = 0; j < feat; ++j) z += w[j] * X[i*feat+j];
        if ((sigmoid(z) >= 0.5f ? 1 : 0) == y[i]) correct++;
    }
    return {ms, loss, 100.0 * correct / n, n, "AdamW"};
}
