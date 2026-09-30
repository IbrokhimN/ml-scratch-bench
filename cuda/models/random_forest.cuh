// Random forest on the GPU.
// Trees are built on the CPU because splitting is recursive and data dependent.
// Prediction runs on the GPU with one thread per sample.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include "../common/cuda_utils.cuh"
#include "../common/rng.hpp"
#include "../common/timer.hpp"

// A tree stored as flat arrays so it can be copied to the GPU.
struct FlatTree {
    std::vector<int>   feature;
    std::vector<float> threshold;
    std::vector<int>   left_child;
    std::vector<int>   right_child;
    std::vector<int>   label;
};

// Host side recursive builder. Uses the same splitting rule as the CPU forest.
static void build_flat_tree(const std::vector<float>& X, const std::vector<int>& y,
                              int n, int feat, int K,
                              std::vector<int>& sample_idx,
                              int max_depth, int min_leaf,
                              FlatTree& tree, int depth) {
    int node_id=(int)tree.feature.size();
    tree.feature.push_back(-1);
    tree.threshold.push_back(0.f);
    tree.left_child.push_back(-1);
    tree.right_child.push_back(-1);
    tree.label.push_back(-1);

    bool is_leaf=false;
    if(depth>=max_depth || (int)sample_idx.size()<=min_leaf) is_leaf=true;
    if(!is_leaf){
        bool pure=true; int first=y[sample_idx[0]];
        for(int i:sample_idx) if(y[i]!=first){pure=false;break;}
        if(pure) is_leaf=true;
    }
    if(is_leaf){
        std::vector<int> cnt(K,0);
        for(int i:sample_idx) cnt[y[i]]++;
        tree.label[node_id]=(int)(std::max_element(cnt.begin(),cnt.end())-cnt.begin());
        return;
    }

    int n_try=std::max(1,(int)sqrtf((float)feat));
    std::vector<int> fs(feat); std::iota(fs.begin(),fs.end(),0);
    std::shuffle(fs.begin(),fs.end(),G_RNG); fs.resize(n_try);

    std::vector<int> cnt_p(K,0);
    for(int i:sample_idx) cnt_p[y[i]]++;
    int tot_p=(int)sample_idx.size();
    float gini_p=1.f;
    for(int k=0;k<K;++k){float p=(float)cnt_p[k]/tot_p; gini_p-=p*p;}

    float best_gain=-1e30f; int best_f=-1; float best_thr=0.f;
    std::vector<int> best_l,best_r;

    for(int f:fs){
        std::vector<std::pair<float,int>> vals;
        for(int i:sample_idx) vals.push_back({X[i*feat+f],i});
        std::sort(vals.begin(),vals.end());
        int sz=(int)vals.size();
        std::vector<int> l_idx,r_idx(sample_idx);
        for(int vi=0;vi<sz-1;++vi){
            l_idx.push_back(vals[vi].second);
            r_idx.erase(std::find(r_idx.begin(),r_idx.end(),vals[vi].second));
            if(vals[vi].first==vals[vi+1].first) continue;
            std::vector<int> lc(K,0); for(int i:l_idx) lc[y[i]]++;
            int nl=(int)l_idx.size(); float gl=1.f;
            for(int k=0;k<K;++k){float p=(float)lc[k]/nl; gl-=p*p;}
            std::vector<int> rc(K,0); for(int i:r_idx) rc[y[i]]++;
            int nr=(int)r_idx.size(); float gr=1.f;
            for(int k=0;k<K;++k){float p=(float)rc[k]/nr; gr-=p*p;}

            float thr=(vals[vi].first+vals[vi+1].first)*0.5f;
            float gain=gini_p-((float)nl/tot_p*gl+(float)nr/tot_p*gr);
            if(gain>best_gain){best_gain=gain;best_f=f;best_thr=thr;best_l=l_idx;best_r=r_idx;}
        }
    }
    if(best_f==-1||best_l.empty()||best_r.empty()){
        std::vector<int> cnt(K,0); for(int i:sample_idx) cnt[y[i]]++;
        tree.label[node_id]=(int)(std::max_element(cnt.begin(),cnt.end())-cnt.begin());
        return;
    }
    tree.feature[node_id]=best_f; tree.threshold[node_id]=best_thr;
    tree.left_child[node_id]=(int)tree.feature.size();
    build_flat_tree(X,y,n,feat,K,best_l,max_depth,min_leaf,tree,depth+1);
    tree.right_child[node_id]=(int)tree.feature.size();
    build_flat_tree(X,y,n,feat,K,best_r,max_depth,min_leaf,tree,depth+1);
}

// One thread per sample. Walks one tree and adds a vote for the predicted class.
__global__ void kernel_rf_predict(
        const float* __restrict__ X,
        const int*   __restrict__ feat_arr,
        const float* __restrict__ thr_arr,
        const int*   __restrict__ left_arr,
        const int*   __restrict__ right_arr,
        const int*   __restrict__ lbl_arr,
        int* __restrict__ votes,
        int n, int feat, int K) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int node=0;
    while(feat_arr[node]!=-1){
        if(X[i*feat+feat_arr[node]]<=thr_arr[node]) node=left_arr[node];
        else                                          node=right_arr[node];
    }
    atomicAdd(&votes[i*K+lbl_arr[node]],1);
}

// Result of a forest run.
struct RFResult { double time_ms,accuracy; int n_samples; };

// Builds the trees on the host, then predicts with one kernel per tree.
// Reported time is build time plus predict time.
inline RFResult rf_cuda(const std::vector<float>& hX, const std::vector<int>& hy,
                         int n, int feat, int K,
                         int n_trees=20, int max_depth=8, int min_leaf=4) {
    std::vector<FlatTree> forest(n_trees);
    auto t_build=Clock::now();

    for(int t=0;t<n_trees;++t){
        std::vector<int> bag(n);
        for(int i=0;i<n;++i) bag[i]=randi(0,n);
        build_flat_tree(hX,hy,n,feat,K,bag,max_depth,min_leaf,forest[t],0);
    }
    double build_ms=Ms(Clock::now()-t_build).count();

    float* dX;
    CUDA_CHECK(cudaMalloc(&dX,n*feat*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX,hX.data(),n*feat*sizeof(float),cudaMemcpyHostToDevice));

    int* d_votes;
    CUDA_CHECK(cudaMalloc(&d_votes,n*K*sizeof(int)));
    CUDA_CHECK(cudaMemset(d_votes,0,n*K*sizeof(int)));

    auto t0=Clock::now();

    for(int t=0;t<n_trees;++t){
        const FlatTree& tree=forest[t];
        int sz=(int)tree.feature.size();

        int *d_feat,*d_left,*d_right,*d_lbl;
        float *d_thr;
        CUDA_CHECK(cudaMalloc(&d_feat,  sz*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_thr,   sz*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_left,  sz*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_right, sz*sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_lbl,   sz*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_feat, tree.feature.data(),    sz*sizeof(int),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_thr,  tree.threshold.data(),  sz*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_left, tree.left_child.data(), sz*sizeof(int),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_right,tree.right_child.data(),sz*sizeof(int),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lbl,  tree.label.data(),      sz*sizeof(int),  cudaMemcpyHostToDevice));

        kernel_rf_predict<<<(n+255)/256,256>>>(dX,d_feat,d_thr,d_left,d_right,d_lbl,d_votes,n,feat,K);

        cudaFree(d_feat); cudaFree(d_thr); cudaFree(d_left);
        cudaFree(d_right); cudaFree(d_lbl);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double predict_ms=Ms(Clock::now()-t0).count();

    std::vector<int> h_votes(n*K);
    CUDA_CHECK(cudaMemcpy(h_votes.data(),d_votes,n*K*sizeof(int),cudaMemcpyDeviceToHost));
    int correct=0;
    for(int i=0;i<n;++i){
        int pred=0;
        for(int k=1;k<K;++k) if(h_votes[i*K+k]>h_votes[i*K+pred]) pred=k;
        if(pred==hy[i]) correct++;
    }
    cudaFree(dX); cudaFree(d_votes);
    return {build_ms+predict_ms, 100.0*correct/n, n};
}
