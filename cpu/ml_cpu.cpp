// ml_cpu.cpp — ML algos from scratch, c++17, no deps
// compile: g++ -std=c++17 -O3 -march=native -fopenmp -o ml_cpu ml_cpu.cpp -lm

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cmath>
#include <cstring>
#include <cassert>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>
#include <string>

static std::mt19937_64 G_RNG(0xDEADBEEFCAFE1234ULL);

inline float randf()  { return std::uniform_real_distribution<float>(0.f, 1.f)(G_RNG); }
inline float randnf() { return std::normal_distribution<float>(0.f, 1.f)(G_RNG); }
inline int   randi(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi-1)(G_RNG); }

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// data gen
void gen_regression(std::vector<float>& X, std::vector<float>& y, int n, int feat) {
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

void gen_classification(std::vector<float>& X, std::vector<int>& y, int n, int feat) {
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

void gen_multiclass(std::vector<float>& X, std::vector<int>& y, int n, int feat, int K) {
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

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float relu(float x)    { return x > 0.f ? x : 0.f; }
inline float relu_d(float x)  { return x > 0.f ? 1.f : 0.f; }
inline float tanh_act(float x){ return std::tanh(x); }
inline float tanh_d(float x)  { float t = std::tanh(x); return 1.f - t*t; }

void softmax_inplace(float* v, int K) {
    float mx = *std::max_element(v, v+K);
    float sum = 0.f;
    for (int k = 0; k < K; ++k) { v[k] = std::exp(v[k] - mx); sum += v[k]; }
    for (int k = 0; k < K; ++k)   v[k] /= sum;
}

inline float mse(const float* pred, const float* target, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) { float d = pred[i]-target[i]; s += d*d; }
    return s / n;
}

float cross_entropy(const std::vector<float>& logits, const std::vector<int>& labels, int n, int K) {
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

// adamw
struct AdamWResult {
    double time_ms;
    double final_loss;
    double accuracy;
    int    n_samples;
    std::string tag;
};

AdamWResult adamw_logistic(const std::vector<float>& X, const std::vector<int>& y,
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

// nadam
AdamWResult nadam_logistic(const std::vector<float>& X, const std::vector<int>& y,
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

// rmsprop
AdamWResult rmsprop_logistic(const std::vector<float>& X, const std::vector<int>& y,
                              int n, int feat, int epochs, int batch_size,
                              float lr=1e-3f, float rho=0.9f,
                              float eps=1e-8f, float momentum=0.9f) {
    std::vector<float> w(feat,0.f), eg2(feat,0.f), delta_w(feat,0.f);
    float b=0.f, eg2_b=0.f, delta_b=0.f;
    double loss=0.0;
    std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0);

    auto t0 = Clock::now();

    for(int ep=0;ep<epochs;++ep){
        std::shuffle(idx.begin(),idx.end(),G_RNG);
        loss=0.0;
        for(int start=0;start<n;start+=batch_size){
            int end=std::min(start+batch_size,n), bs=end-start;
            std::vector<float> gw(feat,0.f); float gb=0.f,bl=0.f;
            for(int ii=start;ii<end;++ii){
                int i=idx[ii];
                float z=b; for(int j=0;j<feat;++j) z+=w[j]*X[i*feat+j];
                float p=sigmoid(z), yi=(float)y[i];
                bl+=-(yi*std::log(p+1e-9f)+(1.f-yi)*std::log(1.f-p+1e-9f));
                float dz=p-yi;
                for(int j=0;j<feat;++j) gw[j]+=dz*X[i*feat+j];
                gb+=dz;
            }
            for(int j=0;j<feat;++j) gw[j]/=bs; gb/=bs;
            loss+=bl/bs;
            for(int j=0;j<feat;++j){
                eg2[j] = rho*eg2[j] + (1.f-rho)*gw[j]*gw[j];
                delta_w[j] = momentum*delta_w[j] - lr * gw[j] / (std::sqrt(eg2[j]) + eps);
                w[j] += delta_w[j];
            }
            eg2_b   = rho*eg2_b   + (1.f-rho)*gb*gb;
            delta_b = momentum*delta_b - lr*gb/(std::sqrt(eg2_b)+eps);
            b      += delta_b;
        }
        loss/=(double)(n/batch_size+1);
    }
    double ms=Ms(Clock::now()-t0).count();
    int correct=0;
    for(int i=0;i<n;++i){
        float z=b; for(int j=0;j<feat;++j) z+=w[j]*X[i*feat+j];
        if((sigmoid(z)>=0.5f?1:0)==y[i]) correct++;
    }
    return {ms,loss,100.0*correct/n,n,"RMSProp"};
}

// sgd + nesterov
AdamWResult sgd_nesterov_logistic(const std::vector<float>& X, const std::vector<int>& y,
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

// sgdr — cosine warm restarts
AdamWResult sgdr_logistic(const std::vector<float>& X, const std::vector<int>& y,
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

// lbfgs
struct LBFGSResult {
    double time_ms, final_loss, accuracy;
    int    n_samples;
};

static float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float s=0.f;
    for(size_t i=0;i<a.size();++i) s+=a[i]*b[i];
    return s;
}

static double bce_loss_grad(const std::vector<float>& X, const std::vector<int>& y,
                             const std::vector<float>& theta,
                             int n, int feat, std::vector<float>& grad) {
    grad.assign(feat+1, 0.f);
    double loss=0.0;
    for(int i=0;i<n;++i){
        float z=theta[feat];
        for(int j=0;j<feat;++j) z+=theta[j]*X[i*feat+j];
        float p=sigmoid(z), yi=(float)y[i];
        loss+=-(yi*std::log(p+1e-9f)+(1.f-yi)*std::log(1.f-p+1e-9f));
        float dz=p-yi;
        for(int j=0;j<feat;++j) grad[j]+=dz*X[i*feat+j];
        grad[feat]+=dz;
    }
    for(auto& g:grad) g/=n;
    return loss/n;
}

LBFGSResult lbfgs_logistic(const std::vector<float>& X, const std::vector<int>& y,
                             int n, int feat, int max_iter=100, int M=10) {
    int d=feat+1;
    std::vector<float> theta(d, 0.f);
    std::vector<float> grad(d, 0.f);

    std::vector<std::vector<float>> s_hist(M, std::vector<float>(d,0.f));
    std::vector<std::vector<float>> y_hist(M, std::vector<float>(d,0.f));
    std::vector<float> rho_hist(M, 0.f);
    int history_len=0, head=0;

    auto t0=Clock::now();

    double loss=bce_loss_grad(X,y,theta,n,feat,grad);

    for(int iter=0;iter<max_iter;++iter){
        // two-loop recursion
        std::vector<float> q=grad;
        std::vector<float> alpha_arr(M,0.f);

        for(int i=history_len-1;i>=0;--i){
            int idx=(head-1-i+2*M)%M;
            float ai=rho_hist[idx]*dot(s_hist[idx],q);
            alpha_arr[i]=ai;
            for(int k=0;k<d;++k) q[k]-=ai*y_hist[idx][k];
        }

        float gamma=1.f;
        if(history_len>0){
            int last=(head-1+M)%M;
            float sy=dot(s_hist[last],y_hist[last]);
            float yy=dot(y_hist[last],y_hist[last]);
            if(yy>1e-10f) gamma=sy/yy;
        }
        std::vector<float> r(d);
        for(int k=0;k<d;++k) r[k]=gamma*q[k];

        for(int i=0;i<history_len;++i){
            int idx=(head-history_len+i+2*M)%M;
            float beta=rho_hist[idx]*dot(y_hist[idx],r);
            for(int k=0;k<d;++k) r[k]+=s_hist[idx][k]*(alpha_arr[i]-beta);
        }

        std::vector<float> p(d);
        for(int k=0;k<d;++k) p[k]=-r[k];

        // armijo line search
        float step=1.0f;
        float c1=1e-4f;
        float pg=dot(p,grad);
        std::vector<float> theta_new(d), grad_new(d);
        double loss_new;
        int ls_iter=0;
        do {
            for(int k=0;k<d;++k) theta_new[k]=theta[k]+step*p[k];
            loss_new=bce_loss_grad(X,y,theta_new,n,feat,grad_new);
            if(loss_new <= loss + c1*step*pg) break;
            step*=0.5f;
        } while(++ls_iter<30 && step>1e-10f);

        auto& s_new=s_hist[head];
        auto& y_new=y_hist[head];
        for(int k=0;k<d;++k){ s_new[k]=theta_new[k]-theta[k]; y_new[k]=grad_new[k]-grad[k]; }
        float sy=dot(s_new,y_new);
        rho_hist[head]=(sy>1e-10f) ? 1.f/sy : 0.f;
        head=(head+1)%M;
        if(history_len<M) history_len++;

        theta=theta_new;
        grad=grad_new;
        loss=loss_new;

        float gnorm=0.f;
        for(float g:grad) gnorm=std::max(gnorm,std::abs(g));
        if(gnorm<1e-5f) break;
    }

    double ms=Ms(Clock::now()-t0).count();
    int correct=0;
    for(int i=0;i<n;++i){
        float z=theta[feat]; for(int j=0;j<feat;++j) z+=theta[j]*X[i*feat+j];
        if((sigmoid(z)>=0.5f?1:0)==y[i]) correct++;
    }
    return {ms, loss, 100.0*correct/n, n};
}

// gmm em
struct GMMResult { double time_ms, log_likelihood; int n_samples, K; };

static float log_gaussian(const float* x, const float* mu, const float* var, int d) {
    float lp = 0.f;
    const float LOG_2PI = 1.8378770664093455f;
    for(int j=0;j<d;++j){
        float diff = x[j]-mu[j];
        lp += -0.5f*(LOG_2PI + std::log(var[j]+1e-6f) + diff*diff/(var[j]+1e-6f));
    }
    return lp;
}

GMMResult gmm_em(const std::vector<float>& X, int n, int feat, int K, int max_iter=100) {
    std::vector<float> mu(K*feat), var(K*feat, 1.f), pi(K, 1.f/K);

    // kmeans++ init
    std::vector<int> chosen;
    chosen.push_back(randi(0,n));
    for(int c=1;c<K;++c){
        std::vector<float> dist2(n, 1e30f);
        for(int i=0;i<n;++i)
            for(int prev:chosen){
                float d=0.f;
                for(int j=0;j<feat;++j){float dd=X[i*feat+j]-X[prev*feat+j];d+=dd*dd;}
                dist2[i]=std::min(dist2[i],d);
            }
        float total=0.f; for(float v:dist2) total+=v;
        float r=randf()*total, cum=0.f;
        int pick=n-1;
        for(int i=0;i<n;++i){ cum+=dist2[i]; if(cum>=r){pick=i;break;} }
        chosen.push_back(pick);
    }
    for(int c=0;c<K;++c)
        for(int j=0;j<feat;++j) mu[c*feat+j]=X[chosen[c]*feat+j];

    std::vector<float> resp(n*K);
    auto t0=Clock::now();

    double log_lik=0.0;
    for(int iter=0;iter<max_iter;++iter){
        // e-step
        log_lik=0.0;
        for(int i=0;i<n;++i){
            float mx=-1e30f;
            std::vector<float> log_p(K);
            for(int k=0;k<K;++k){
                log_p[k] = std::log(pi[k]+1e-9f) + log_gaussian(&X[i*feat],&mu[k*feat],&var[k*feat],feat);
                mx=std::max(mx,log_p[k]);
            }
            float sum_exp=0.f;
            for(int k=0;k<K;++k){ resp[i*K+k]=std::exp(log_p[k]-mx); sum_exp+=resp[i*K+k]; }
            log_lik+=std::log(sum_exp)+mx;
            for(int k=0;k<K;++k) resp[i*K+k]/=sum_exp;
        }
        log_lik/=n;

        // m-step
        std::vector<float> Nk(K,0.f);
        std::vector<float> new_mu(K*feat,0.f), new_var(K*feat,0.f);
        for(int i=0;i<n;++i)
            for(int k=0;k<K;++k){
                float r=resp[i*K+k];
                Nk[k]+=r;
                for(int j=0;j<feat;++j) new_mu[k*feat+j]+=r*X[i*feat+j];
            }
        for(int k=0;k<K;++k){
            float nk=std::max(Nk[k],1e-6f);
            pi[k]=nk/n;
            for(int j=0;j<feat;++j) new_mu[k*feat+j]/=nk;
        }
        for(int i=0;i<n;++i)
            for(int k=0;k<K;++k){
                float r=resp[i*K+k];
                for(int j=0;j<feat;++j){
                    float d=X[i*feat+j]-new_mu[k*feat+j];
                    new_var[k*feat+j]+=r*d*d;
                }
            }
        for(int k=0;k<K;++k){
            float nk=std::max(Nk[k],1e-6f);
            for(int j=0;j<feat;++j) new_var[k*feat+j]=new_var[k*feat+j]/nk + 1e-4f;
        }
        mu=new_mu; var=new_var;
    }

    double ms=Ms(Clock::now()-t0).count();
    return {ms, log_lik, n, K};
}

// kernel pca
struct KPCAResult { double time_ms, variance_explained; int n_samples; };

KPCAResult kernel_pca(const std::vector<float>& X, int n, int feat,
                      int m_landmarks=64, float sigma2=1.0f, int power_iter=20) {
    std::vector<int> lm(n); std::iota(lm.begin(),lm.end(),0);
    std::shuffle(lm.begin(),lm.end(),G_RNG);
    lm.resize(m_landmarks);

    auto rbf=[&](const float* a, const float* b)->float{
        float d=0.f;
        for(int j=0;j<feat;++j){float dd=a[j]-b[j]; d+=dd*dd;}
        return std::exp(-d/(2.f*sigma2));
    };

    // build Kmm
    std::vector<float> Kmm(m_landmarks*m_landmarks);
    for(int i=0;i<m_landmarks;++i)
        for(int j=0;j<m_landmarks;++j)
            Kmm[i*m_landmarks+j]=rbf(&X[lm[i]*feat],&X[lm[j]*feat]);

    // build Knm
    std::vector<float> Knm(n*m_landmarks);
    for(int i=0;i<n;++i)
        for(int j=0;j<m_landmarks;++j)
            Knm[i*m_landmarks+j]=rbf(&X[i*feat],&X[lm[j]*feat]);

    auto t0=Clock::now();

    std::vector<float> row_mean_Knm(n,0.f), col_mean_Knm(m_landmarks,0.f);
    float global_mean_Kmm=0.f;
    for(int i=0;i<n;++i)
        for(int j=0;j<m_landmarks;++j) row_mean_Knm[i]+=Knm[i*m_landmarks+j];
    for(int i=0;i<n;++i) row_mean_Knm[i]/=m_landmarks;
    for(int j=0;j<m_landmarks;++j)
        for(int i=0;i<n;++i) col_mean_Knm[j]+=Knm[i*m_landmarks+j];
    for(int j=0;j<m_landmarks;++j) col_mean_Knm[j]/=n;
    for(int v:lm)
        for(int j=0;j<m_landmarks;++j) global_mean_Kmm+=Kmm[0*m_landmarks+j];
    global_mean_Kmm/=(m_landmarks*m_landmarks);

    std::vector<float> Knm_c(n*m_landmarks);
    for(int i=0;i<n;++i)
        for(int j=0;j<m_landmarks;++j)
            Knm_c[i*m_landmarks+j] = Knm[i*m_landmarks+j]
                                    - row_mean_Knm[i]
                                    - col_mean_Knm[j]
                                    + global_mean_Kmm;

    // power iteration
    std::vector<float> v(n, 1.f/std::sqrt((float)n));
    float eigenval=0.f;
    for(int it=0;it<power_iter;++it){
        std::vector<float> Ktv(m_landmarks,0.f);
        for(int i=0;i<n;++i)
            for(int j=0;j<m_landmarks;++j) Ktv[j]+=Knm_c[i*m_landmarks+j]*v[i];
        std::vector<float> Kw(n,0.f);
        for(int i=0;i<n;++i)
            for(int j=0;j<m_landmarks;++j) Kw[i]+=Knm_c[i*m_landmarks+j]*Ktv[j];
        eigenval=0.f; for(float x:Kw) eigenval+=x*x; eigenval=std::sqrt(eigenval);
        if(eigenval<1e-10f) break;
        for(int i=0;i<n;++i) v[i]=Kw[i]/eigenval;
    }

    float trace=0.f;
    for(int i=0;i<n;++i) for(int j=0;j<m_landmarks;++j)
        trace+=Knm_c[i*m_landmarks+j]*Knm_c[i*m_landmarks+j];
    float var_expl = (trace>0.f) ? (eigenval/(trace/m_landmarks))*100.f : 0.f;

    double ms=Ms(Clock::now()-t0).count();
    return {ms, (double)var_expl, n};
}

// mlp adamw
struct MLPResult { double time_ms, final_loss, accuracy; int n_samples; };

static void he_init(std::vector<float>& W, int fan_in, int fan_out) {
    float std = std::sqrt(2.f / fan_in);
    W.resize(fan_out * fan_in);
    for(auto& v:W) v=randnf()*std;
}

MLPResult mlp_adamw(const std::vector<float>& X, const std::vector<int>& y,
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

// random forest
struct RFResult { double time_ms, accuracy, oob_accuracy; int n_samples; };

struct TreeNode {
    int    feature   = -1;
    float  threshold = 0.f;
    int    left      = -1;
    int    right     = -1;
    int    label     = -1;
};

static float gini(const std::vector<int>& labels, const std::vector<int>& indices, int K) {
    std::vector<int> cnt(K, 0);
    int total = (int)indices.size();
    if(total==0) return 0.f;
    for(int i:indices) cnt[labels[i]]++;
    float g=1.f;
    for(int k=0;k<K;++k){ float p=(float)cnt[k]/total; g-=p*p; }
    return g;
}

static int majority_vote(const std::vector<int>& labels, const std::vector<int>& indices, int K) {
    std::vector<int> cnt(K,0);
    for(int i:indices) cnt[labels[i]]++;
    return (int)(std::max_element(cnt.begin(),cnt.end())-cnt.begin());
}

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

static int predict_tree(const std::vector<TreeNode>& nodes, const float* x, int feat, int node_idx=0){
    const TreeNode& node=nodes[node_idx];
    if(node.feature==-1) return node.label;
    if(x[node.feature]<=node.threshold) return predict_tree(nodes,x,feat,node.left);
    else                                 return predict_tree(nodes,x,feat,node.right);
}

RFResult random_forest(const std::vector<float>& X, const std::vector<int>& y,
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

static std::ofstream G_CSV;
static void write_row(const std::string& algo, int n, int feat,
                      double time_ms, const std::string& mname, double mval) {
    G_CSV << algo << "," << n << "," << feat << "," << time_ms
          << "," << mname << "," << mval << ",CPU\n";
}

int main(int argc, char* argv[]) {
    std::string out = (argc>1) ? argv[1] : "results/cpu_results.csv";
    G_CSV.open(out);
    if(!G_CSV){ std::cerr<<"Cannot open "<<out<<"\n"; return 1; }
    G_CSV << "algorithm,n_samples,n_features,time_ms,metric_name,metric_value,device\n";
    G_CSV << std::fixed;

    std::vector<int> sizes = {512, 1024, 2048, 4096, 8192, 16384};
    const int FEAT   = 32;
    const int EPOCHS = 60;
    const int BATCH  = 64;
    const int K_CLS  = 4;

    for(int n : sizes) {
        std::cout << "[CPU] n=" << n << "\n";

        std::vector<float> X_reg; std::vector<float> y_reg;
        std::vector<float> X_cls; std::vector<int>   y_cls;
        gen_regression(X_reg, y_reg, n, FEAT);
        gen_multiclass(X_cls, y_cls, n, FEAT, K_CLS);

        std::vector<int> y_bin(n);
        for(int i=0;i<n;++i) y_bin[i]=(y_cls[i]%2);

        std::cout << "  AdamW ... " << std::flush;
        { auto r=adamw_logistic(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("AdamW",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("AdamW",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms, acc=" << r.accuracy << "%\n"; }

        std::cout << "  Nadam ... " << std::flush;
        { auto r=nadam_logistic(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("Nadam",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("Nadam",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  RMSProp ... " << std::flush;
        { auto r=rmsprop_logistic(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("RMSProp",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("RMSProp",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  SGD_Nesterov ... " << std::flush;
        { auto r=sgd_nesterov_logistic(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("SGD_Nesterov",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("SGD_Nesterov",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  SGDR ... " << std::flush;
        { auto r=sgdr_logistic(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("SGDR",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("SGDR",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  L-BFGS ... " << std::flush;
        { auto r=lbfgs_logistic(X_cls,y_bin,n,FEAT,50,10);
          write_row("LBFGS",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("LBFGS",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  GMM-EM ... " << std::flush;
        { auto r=gmm_em(X_cls,n,FEAT,K_CLS,80);
          write_row("GMM_EM",n,FEAT,r.time_ms,"LogLikelihood",r.log_likelihood);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  Kernel PCA ... " << std::flush;
        { auto r=kernel_pca(X_cls,n,FEAT,std::min(n,128),1.0f,30);
          write_row("KernelPCA",n,FEAT,r.time_ms,"VarExplained",r.variance_explained);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  MLP-AdamW ... " << std::flush;
        { auto r=mlp_adamw(X_cls,y_cls,n,FEAT,K_CLS,EPOCHS,BATCH);
          write_row("MLP_AdamW",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("MLP_AdamW",n,FEAT,r.time_ms,"CE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  RandomForest ... " << std::flush;
        { int n_trees=20, mxd=8;
          int n_rf = std::min(n, 4096);
          std::vector<float> X_rf(X_cls.begin(), X_cls.begin()+n_rf*FEAT);
          std::vector<int>   y_rf(y_cls.begin(), y_cls.begin()+n_rf);
          auto r=random_forest(X_rf,y_rf,n_rf,FEAT,K_CLS,n_trees,mxd,4);
          write_row("RandomForest",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("RandomForest",n,FEAT,r.time_ms,"OOB_Accuracy",r.oob_accuracy);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << std::flush;
    }

    G_CSV.close();
    std::cout << "\n[CPU] Results written to " << out << "\n";
    return 0;
}
