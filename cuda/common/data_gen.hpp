// Synthetic dataset generator.
#pragma once

#include <vector>

#include "rng.hpp"

// K gaussian clusters, one per class. Same generator as the CPU build.
inline void gen_multiclass(std::vector<float>& X, std::vector<int>& y, int n, int feat, int K) {
    X.resize(n*feat); y.resize(n);
    std::vector<float> centres(K*feat);
    for(auto& v:centres) v=randnf()*2.f;
    for(int i=0;i<n;++i){
        int cls=randi(0,K); y[i]=cls;
        for(int j=0;j<feat;++j) X[i*feat+j]=centres[cls*feat+j]+randnf()*0.8f;
    }
}
