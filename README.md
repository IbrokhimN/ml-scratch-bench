# From-Scratch ML Benchmark: CPU vs CUDA GPU

> *No cuDNN. No cuBLAS. No PyTorch. Ten algorithms written from zero in C++17 and raw CUDA kernels.*

**Technical Report · ML Systems · 2025**  
`d=32 features` · `Epochs=60` · `Batch=64` · `K=4 classes` · `Seed=0xDEADBEEFCAFE1234`

---

## Abstract

We benchmark ten classical machine learning algorithms implemented entirely from scratch in C++17 (CPU) and CUDA C++ (GPU), with zero high-level library dependencies. All GPU kernels are hand-written — forward passes, gradient computation, optimizer updates, EM steps, RBF matrix construction, tree prediction. Experiments span six dataset scales (N = 512 to 16,384) with d = 32 features. GPU acceleration ranges from **0.33× (Random Forest, N=512)** to **24.7× (MLP-AdamW, N=16,384)**, with breakeven points between N=512 and N=4,096 depending on algorithm class. We identify three distinct scaling regimes and characterize the L-BFGS anomaly as a consequence of adaptive convergence behavior rather than numerical issues.

---

## Table of Contents

1. [Setup](#1-setup)
2. [Results](#2-results)  
   2.1 [Benchmark Figures](#21-benchmark-figures)  
   2.2 [Raw Timing at N=16,384](#22-raw-timing-at-n16384)  
   2.3 [Speedup Across All N](#23-speedup-across-all-n)  
   2.4 [Breakeven Points](#24-breakeven-points)
3. [Analysis](#3-analysis)  
   3.1 [Three Scaling Regimes](#31-three-scaling-regimes)  
   3.2 [L-BFGS: The Anomaly](#32-l-bfgs-the-anomaly)  
   3.3 [Convergence Parity](#33-convergence-parity)
4. [Implementation Notes](#4-implementation-notes)
5. [Reproduce](#5-reproduce)

---

## 1. Setup

### Hardware & Compilation

| | Spec |
|---|---|
| **CPU flags** | `-std=c++17 -O3 -march=native -fopenmp` |
| **GPU flags** | `-std=c++17 -O3 -arch=sm_75 -use_fast_math` |
| **C++ standard** | C++17 |
| **Dependencies** | **None** (stdlib + CUDA runtime only) |

### Data

Synthetic, fixed seed `0xDEADBEEFCAFE1234`. Multiclass: K=4 Gaussian clusters, σ=0.8, centroid spread=2.0. Binary: multiclass labels mod 2. All experiments use d=32 features, 60 epochs, batch=64, across N ∈ {512, 1024, 2048, 4096, 8192, 16384}.

### Algorithms

| Algorithm | Category | GPU Strategy |
|---|---|---|
| `AdamW` | First-order optimizer | Per-sample fwd + per-feature grad + Adam update kernel |
| `Nadam` | First-order optimizer | Nesterov correction folded into update kernel |
| `RMSProp` | First-order optimizer | Parallel EG² accumulation + momentum delta kernel |
| `SGD+Nesterov` | First-order optimizer | Lookahead kernel → grad on lookahead → Nesterov step |
| `SGDR` | First-order optimizer | Cosine warm restarts, standard SGD update kernel |
| `L-BFGS` | Second-order optimizer | GPU two-loop recursion, GPU dot products |
| `MLP (AdamW)` | Neural network | Full fwd/bwd kernels, He init, 64→32→K, per-layer AdamW |
| `GMM-EM` | Unsupervised | Parallel E-step (per-sample), M-step via atomics |
| `Kernel PCA` | Dim. reduction | 2D RBF matrix kernel + power iteration on device |
| `Random Forest` | Ensemble | CPU tree build, GPU parallel prediction per sample |

---

## 2. Results

### 2.1 Benchmark Figures

![Full benchmark results — CPU vs CUDA GPU across all algorithms and sample sizes](https://raw.githubusercontent.com/IbrokhimN/ml-scratch-bench/refs/heads/main/results/benchmark_plots_v3.png)


**Figure 1.** Complete benchmark results.
*Top-left:* Absolute execution time (ms) at N=16,384 for CPU (orange) and CUDA (green).
*Top-right:* Time vs N in log scale; dashed = CPU, solid = CUDA.
*Middle rows:* Per-algorithm speedup curves with breakeven markers.
*Bottom-left:* Speedup heatmap across N × algorithm; red = GPU overhead, green = GPU faster.
*Bottom-center:* Total cumulative compute time.
*Bottom-right:* Peak speedup per algorithm at N=16,384, sorted ascending.

---

### 2.2 Raw Timing at N=16,384

| Algorithm | CPU (ms) | CUDA (ms) | Speedup | Category |
|---|---:|---:|---:|---|
| **MLP (AdamW)** | 8,434 | 342 | **24.7×** | Neural network |
| **Kernel PCA** | 298 | 16 | **18.5×** | Dim. reduction |
| **GMM-EM** | 1,946 | 123 | **15.9×** | Unsupervised |
| AdamW | 249 | 31 | 8.0× | First-order opt. |
| Nadam | 286 | 40 | 7.1× | First-order opt. |
| RMSProp | 280 | 40 | 7.1× | First-order opt. |
| SGD+Nesterov | 276 | 42 | 6.6× | First-order opt. |
| SGDR | 224 | 35 | 6.4× | First-order opt. |
| L-BFGS ⚠ | 824 | 134 | 6.1× | Second-order opt. |
| Random Forest | 3,985 | 1,356 | 2.9× | Ensemble |

> ⚠ L-BFGS uses adaptive convergence; speedup is irregular across N — see [§3.2](#32-l-bfgs-the-anomaly).

---

### 2.3 Speedup Across All N

| Algorithm | N=512 | N=1k | N=2k | N=4k | N=8k | N=16k | Trend |
|---|---:|---:|---:|---:|---:|---:|---|
| AdamW | 1.01 | 1.95 | 3.20 | 4.39 | 6.34 | **7.95** | Monotone ↑ |
| GMM-EM | 4.44 | 7.03 | 9.50 | 12.74 | 14.43 | **15.88** | Monotone ↑ |
| Kernel PCA | 2.25 | 3.00 | 5.97 | 10.14 | 14.87 | **18.47** | Monotone ↑ |
| MLP (AdamW) | 5.37 | 8.09 | 13.60 | 18.31 | 23.71 | **24.69** | Monotone ↑ |
| Nadam | 0.92 | 1.76 | 2.85 | 4.31 | 5.45 | **7.10** | Monotone ↑ |
| RMSProp | 0.91 | 1.68 | 2.70 | 4.23 | 5.80 | **7.08** | Monotone ↑ |
| SGD+Nesterov | 0.80 | 1.54 | 2.48 | 4.05 | 4.97 | **6.63** | Monotone ↑ |
| SGDR | 0.71 | 1.31 | 2.30 | 3.05 | 4.34 | **6.37** | Monotone ↑ |
| L-BFGS | 0.68 | 2.06 | **0.29** | 3.32 | **0.87** | 6.14 | ⚠ Irregular |
| Random Forest | 0.33 | 0.53 | 0.98 | 1.68 | 2.24 | **2.94** | Slow growth |

All 9 non-anomalous algorithms show strictly monotone speedup growth. The speedup curves are still rising at N=16,384 — throughput saturation not yet reached for any algorithm at these parameters.

---

### 2.4 Breakeven Points

The smallest N where CUDA first exceeds CPU (speedup ≥ 1.0):

| Algorithm | Breakeven | Notes |
|---|---|---|
| MLP (AdamW) | < 512 | Dense matmul immediately benefits |
| GMM-EM | < 512 | Per-sample E-step highly parallel |
| Kernel PCA | < 512 | RBF matrix is O(N·M·d), all parallel |
| AdamW | ~512 | Marginally ≥1× at smallest N |
| Nadam | ~1,024 | Slight lookahead overhead |
| RMSProp | ~1,024 | Same pipeline as AdamW |
| SGD+Nesterov | ~1,024 | Extra lookahead kernel adds latency |
| SGDR | ~1,024 | SGD kernel simple but small N still slow |
| L-BFGS | ~1,024 (variable) | Depends on convergence iteration count |
| Random Forest | ~2,048 | Build time dominates at small N |

---

## 3. Analysis

### 3.1 Three Scaling Regimes

**Regime I — Dense compute (MLP, Kernel PCA, GMM-EM): 15–25×**

Dominant operations are O(N·d) or O(N·M) inner products mapping cleanly to 2D thread grids. The GPU's arithmetic throughput dominates over latency. Speedup curves are monotone and still climbing at N=16,384 — peak throughput not yet reached at d=32.

- MLP: 8,434ms CPU → 342ms GPU. A full 60-epoch manual backprop run fits in a third of a second on GPU. Layer-wise fwd/bwd kernels expose massive parallelism across both samples and weights simultaneously.
- Kernel PCA: most dramatic per-ms: 298ms → 16ms. The O(N×M) RBF distance matrix construction is embarrassingly parallel with no reductions. Power iteration runs entirely on device.
- GMM-EM: the E-step computes K log-probabilities per sample independently, a perfect map operation. M-step atomics cause some contention at small K but scale well with N.

**Regime II — Gradient-based optimizers (AdamW, Nadam, RMSProp, SGD, SGDR): 6–8×**

Two-kernel pipeline per step: (1) per-sample forward, (2) per-feature gradient accumulation. Speedup bounded by the reduction step (bias gradient via `gpu_sum`) and by d=32 limiting per-feature parallelism. All five optimizers converge to the same ~7× ceiling at N=16k, consistent with shared architectural bottleneck. Increasing d would raise this ceiling.

**Regime III — Branchy control flow (Random Forest): 2–3×**

Tree traversal is a pointer-chasing loop with data-dependent branches. Warp divergence caps utilization. The 2.9× gain at N=16k comes primarily from running 20 trees' predictions concurrently, not fast per-sample traversal. CPU tree build time (sequential, not parallelizable) is included in all reported timings.

---

### 3.2 L-BFGS: The Anomaly

L-BFGS speedup at each N:

```
N=512:   0.68×  (CPU faster)
N=1k:    2.06×
N=2k:    0.29×  ← CPU wins by 3.4×
N=4k:    3.32×
N=8k:    0.87×  ← CPU wins
N=16k:   6.14×
```

This is not numerical noise. L-BFGS terminates when `||∇f||∞ < 1e-5`. The number of gradient evaluations varies entirely with the conditioning of the specific synthetic dataset generated at each N with the fixed seed. At N=2,048, the CPU converges in ~0.4ms — suspiciously fast, consistent with near-perfectly conditioned random data at that random state — while the GPU pays fixed kernel launch overhead regardless of convergence speed.

The practical implication: **L-BFGS GPU speedup should be measured at fixed iteration counts, not convergence**. The 6.1× at N=16k is meaningful; the 0.29× at N=2k is a dataset conditioning artifact.

---

### 3.3 Convergence Parity

CPU and CUDA implementations reach statistically identical accuracy across all N. Both converge to 100% on the synthetic tasks (well-separated Gaussian clusters, linearly separable regime). Random Forest shows minor variance at N=2k–4k (99.95–99.97% accuracy) from bootstrap sampling randomness, not numerical error.

`-use_fast_math` (approximate reciprocal, sqrt, log on GPU) produces no observable accuracy degradation on these tasks. Loss values between CPU and CUDA agree within floating-point rounding.

---

## 4. Implementation Notes

### Gradient Pipeline (First-Order Optimizers)

Three kernels per step:
1. `kernel_logistic_forward` — parallel over N samples, computes `err[i] = σ(w·xᵢ + b) − yᵢ` and per-sample BCE
2. `kernel_grad_weights` — parallel over d features, computes `gw[j] = (1/n)Σ err[i]·X[i,j]`
3. Optimizer update kernel (AdamW / Nadam / RMSProp / Nesterov) — parallel over d features

Bias gradient via `gpu_sum` (parallel reduction → atomicAdd), updated on host.

### MLP

Architecture: d → H1=64 (ReLU) → H2=32 (ReLU) → K=4 (softmax). All activations and delta tensors live in device memory for the full batch N. Pipeline per step:

```
kernel_layer_fwd ×3
→ kernel_softmax_ce_bwd
→ kernel_layer_bwd_delta ×2
→ kernel_weight_grad ×3
→ kernel_bias_grad ×3
→ kernel_adamw_update ×6
```

He initialization on host, uploaded once.

### GMM-EM

K-means++ initialization on host. E-step: one thread per sample computes all K log-responsibilities in registers (K=4 fits in register file), writes normalized `resp[i,k]` and log-likelihood. M-step: atomic accumulation of Nk, new_mu, new_var followed by divide kernels. 80 EM iterations.

### Kernel PCA (Nyström)

M=min(N,128) random landmarks. RBF matrix `Knm[N×M]` built via 2D kernel with 16×16 thread blocks. Power iteration fully on device: `kernel_matTvec` → `kernel_matvec` → `kernel_normalize`, 30 iterations.

### Random Forest (Hybrid)

20 trees, max_depth=8, min_leaf=4, √d features per split. Tree build on CPU (recursive, data-dependent — not parallelizable). Prediction on GPU: structure uploaded as four flat arrays (feature indices, thresholds, left/right child pointers, labels). One kernel per tree, one thread per sample. Vote accumulation via `atomicAdd` on `votes[N×K]`.

### What Was Deliberately Not Optimized

- No shared memory tiling for matmul (would improve MLP significantly)
- No fp16 / mixed precision
- No cuBLAS or cuSPARSE
- No persistent kernels or CUDA graphs
- No L-BFGS history ring buffer on device (separate allocations per vector)

These omissions are intentional — the goal is to measure algorithmic parallelism, not engineering overhead.

---

## 5. Reproduce

```bash
# build
g++ -std=c++17 -O3 -march=native -fopenmp \
    -o ml_cpu cpu/ml_cpu.cpp -lm

nvcc -std=c++17 -O3 -arch=sm_75 -use_fast_math \
     -o ml_cuda cuda/ml_cuda.cu -lm

# run
./ml_cpu  results/cpu_results.csv
./ml_cuda results/cuda_results.csv

# plot
python3 scripts/plot_results.py
```

Output CSV format:

```
algorithm, n_samples, n_features, time_ms, metric_name, metric_value, device
```

### File Structure

```
ml_benchmark_final/
├── cpu/ml_cpu.cpp          # all CPU implementations
├── cuda/ml_cuda.cu         # all CUDA kernel implementations  
├── scripts/
│   ├── run_benchmark.sh
│   └── plot_results.py
├── results/
│   ├── cpu_results.csv
│   ├── cuda_results.csv
│   ├── combined_results.csv
│   ├── speedup_summary.csv
│   └── benchmark_plots_v3.png
├── CMakeLists.txt
└── Makefile
```

---

*MIT License · No external ML dependencies*
