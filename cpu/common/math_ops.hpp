// Small scalar and vector math helpers shared by the CPU algorithms.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

// Activation functions and their derivatives.
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float relu(float x)    { return x > 0.f ? x : 0.f; }
inline float relu_d(float x)  { return x > 0.f ? 1.f : 0.f; }
inline float tanh_act(float x){ return std::tanh(x); }
inline float tanh_d(float x)  { float t = std::tanh(x); return 1.f - t*t; }

// Numerically stable softmax applied in place.
inline void softmax_inplace(float* v, int K) {
    float mx = *std::max_element(v, v+K);
    float sum = 0.f;
    for (int k = 0; k < K; ++k) { v[k] = std::exp(v[k] - mx); sum += v[k]; }
    for (int k = 0; k < K; ++k)   v[k] /= sum;
}

// Mean squared error between two arrays.
inline float mse(const float* pred, const float* target, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) { float d = pred[i]-target[i]; s += d*d; }
    return s / n;
}

// Mean cross entropy computed from raw logits.
inline float cross_entropy(const std::vector<float>& logits, const std::vector<int>& labels, int n, int K) {
    float loss = 0.f;
    for (int i = 0; i < n; ++i) {
        std::vector<float> p(K);
        float mx = *std::max_element(&logits[i*K], &logits[i*K+K]);
        float sum = 0.f;
        for (int k = 0; k < K; ++k) { p[k] = std::exp(logits[i*K+k]-mx); sum += p[k]; }
        loss -= std::log(p[labels[i]] / sum + 1e-9f);
    }
    return loss / n;
}
