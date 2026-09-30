// Synthetic dataset generators.
// All of them draw from the shared generator so runs are reproducible.
#pragma once

#include <vector>

#include "rng.hpp"

// Linear regression data with a little gaussian noise on the target.
inline void gen_regression(std::vector<float>& X, std::vector<float>& y, int n, int feat) {
    X.resize(n * feat);
    y.resize(n);
    std::vector<float> w_true(feat);
    for (auto& v : w_true) v = (randf() * 2.f - 1.f);
    for (int i = 0; i < n; ++i) {
        float yi = 0.f;
        for (int j = 0; j < feat; ++j) {
            X[i*feat+j] = randnf();
            yi += w_true[j] * X[i*feat+j];
        }
        y[i] = yi + randnf() * 0.1f;
    }
}

// Binary labels from a noisy linear decision boundary.
inline void gen_classification(std::vector<float>& X, std::vector<int>& y, int n, int feat) {
    X.resize(n * feat);
    y.resize(n);
    std::vector<float> w_true(feat);
    for (auto& v : w_true) v = (randf() * 2.f - 1.f);
    for (int i = 0; i < n; ++i) {
        float s = 0.f;
        for (int j = 0; j < feat; ++j) {
            X[i*feat+j] = randnf();
            s += w_true[j] * X[i*feat+j];
        }
        y[i] = (s + randnf() * 0.3f) > 0.f ? 1 : 0;
    }
}

// K gaussian clusters, one per class. This is the dataset the benchmark uses.
inline void gen_multiclass(std::vector<float>& X, std::vector<int>& y, int n, int feat, int K) {
    X.resize(n * feat);
    y.resize(n);
    std::vector<float> centres(K * feat);
    for (auto& v : centres) v = randnf() * 2.f;
    for (int i = 0; i < n; ++i) {
        int cls = randi(0, K);
        y[i] = cls;
        for (int j = 0; j < feat; ++j)
            X[i*feat+j] = centres[cls*feat+j] + randnf() * 0.8f;
    }
}
