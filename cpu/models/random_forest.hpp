// Random forest classifier with Gini splits, bootstrap sampling and out of bag scoring.
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include "../common/rng.hpp"
#include "../common/timer.hpp"

// Result of a forest run.
struct RFResult { double time_ms, accuracy, oob_accuracy; int n_samples; };

// One node of a decision tree. A negative feature index marks a leaf.
struct TreeNode {
    int    feature   = -1;
    float  threshold = 0.f;
    int    left      = -1;
    int    right     = -1;
    int    label     = -1;
};

// Gini impurity of a set of samples.
static float gini(const std::vector<int>& labels, const std::vector<int>& indices, int K) {
    std::vector<int> cnt(K, 0);
    int total = (int)indices.size();
    if(total==0) return 0.f;
    for(int i:indices) cnt[labels[i]]++;
    float g=1.f;
    for(int k=0;k<K;++k){ float p=(float)cnt[k]/total; g-=p*p; }
    return g;
}

// Most common label among the given samples.
static int majority_vote(const std::vector<int>& labels, const std::vector<int>& indices, int K) {
    std::vector<int> cnt(K,0);
    for(int i:indices) cnt[labels[i]]++;
    return (int)(std::max_element(cnt.begin(),cnt.end())-cnt.begin());
}

// Recursively grows a tree. Each split tries sqrt of feat random features
// and keeps the threshold with the best Gini gain.
static void build_tree(const std::vector<float>& X, const std::vector<int>& y,
                        int n, int feat, int K,
                        std::vector<int>& sample_indices,
                        int max_depth, int min_leaf,
                        std::vector<TreeNode>& nodes, int depth) {
    int node_idx = (int)nodes.size();
    nodes.push_back({});
    TreeNode& node = nodes.back();

    if(depth>=max_depth || (int)sample_indices.size()<=min_leaf){
        node.label=majority_vote(y,sample_indices,K); return;
    }
    bool pure=true;
    int first=y[sample_indices[0]];
    for(int i:sample_indices) if(y[i]!=first){pure=false;break;}
    if(pure){ node.label=first; return; }

    int n_try = std::max(1,(int)std::sqrt((float)feat));
    std::vector<int> feat_subset(feat); std::iota(feat_subset.begin(),feat_subset.end(),0);
    std::shuffle(feat_subset.begin(),feat_subset.end(),G_RNG);
    feat_subset.resize(n_try);

    float best_gain = -1e30f;
    int   best_feat = -1;
    float best_thr  = 0.f;
    std::vector<int> best_left, best_right;

    float parent_gini = gini(y, sample_indices, K);

    for(int f:feat_subset){
        std::vector<std::pair<float,int>> vals;
        vals.reserve(sample_indices.size());
        for(int i:sample_indices) vals.push_back({X[i*feat+f],i});
        std::sort(vals.begin(),vals.end());

        std::vector<int> left_idx, right_idx(sample_indices);
        int sz=(int)vals.size();
        for(int vi=0;vi<sz-1;++vi){
            left_idx.push_back(vals[vi].second);
            right_idx.erase(std::find(right_idx.begin(),right_idx.end(),vals[vi].second));
            if(vals[vi].first==vals[vi+1].first) continue;

            float thr=(vals[vi].first+vals[vi+1].first)*0.5f;
            int nl=(int)left_idx.size(), nr=(int)right_idx.size();
            int total=nl+nr;
            float split_g = (float)nl/total*gini(y,left_idx,K)
                          + (float)nr/total*gini(y,right_idx,K);
            float gain=parent_gini-split_g;
            if(gain>best_gain){
                best_gain=gain; best_feat=f; best_thr=thr;
                best_left=left_idx; best_right=right_idx;
            }
        }
    }

    if(best_feat==-1 || best_left.empty() || best_right.empty()){
        node.label=majority_vote(y,sample_indices,K); return;
    }

    node.feature=best_feat; node.threshold=best_thr;

    node.left=(int)nodes.size();
    build_tree(X,y,n,feat,K,best_left, max_depth,min_leaf,nodes,depth+1);
    nodes[node_idx].right=(int)nodes.size();
    build_tree(X,y,n,feat,K,best_right,max_depth,min_leaf,nodes,depth+1);
}

// Walks one sample from the root down to a leaf and returns the leaf label.
static int predict_tree(const std::vector<TreeNode>& nodes, const float* x, int feat, int node_idx=0){
    const TreeNode& node=nodes[node_idx];
    if(node.feature==-1) return node.label;
    if(x[node.feature]<=node.threshold) return predict_tree(nodes,x,feat,node.left);
    else                                 return predict_tree(nodes,x,feat,node.right);
}

// Trains n_trees trees on bootstrap samples and reports accuracy and out of bag accuracy.
// Timing covers tree building only.
inline RFResult random_forest(const std::vector<float>& X, const std::vector<int>& y,
                               int n, int feat, int K,
                               int n_trees=20, int max_depth=8, int min_leaf=4) {
    std::vector<std::vector<TreeNode>> forest(n_trees);
    std::vector<std::vector<int>> oob_samples(n);

    auto t0=Clock::now();

    for(int t=0;t<n_trees;++t){
        std::vector<int> bag(n), oob_mask(n,1);
        for(int i=0;i<n;++i){ int s=randi(0,n); bag[i]=s; oob_mask[s]=0; }
        for(int i=0;i<n;++i) if(oob_mask[i]) oob_samples[i].push_back(t);
        build_tree(X,y,n,feat,K,bag,max_depth,min_leaf,forest[t],0);
    }

    double ms=Ms(Clock::now()-t0).count();

    int correct=0;
    for(int i=0;i<n;++i){
        std::vector<int> votes(K,0);
        for(int t=0;t<n_trees;++t) votes[predict_tree(forest[t],&X[i*feat],feat)]++;
        int pred=(int)(std::max_element(votes.begin(),votes.end())-votes.begin());
        if(pred==y[i]) correct++;
    }
    double acc=100.0*correct/n;

    int oob_correct=0, oob_total=0;
    for(int i=0;i<n;++i){
        if(oob_samples[i].empty()) continue;
        std::vector<int> votes(K,0);
        for(int t:oob_samples[i]) votes[predict_tree(forest[t],&X[i*feat],feat)]++;
        int pred=(int)(std::max_element(votes.begin(),votes.end())-votes.begin());
        if(pred==y[i]) oob_correct++;
        oob_total++;
    }
    double oob_acc = oob_total>0 ? 100.0*oob_correct/oob_total : 0.0;
    return {ms, acc, oob_acc, n};
}
