# ===========================================================
#  Makefile  –  ml_benchmark  (CPU + optional CUDA)
#
#  Targets:
#    make all          – build CPU binary (+ CUDA if nvcc found)
#    make cpu          – CPU only
#    make cuda         – CUDA only
#    make run          – build + run full benchmark + plot
#    make run_cpu      – build + run CPU-only benchmark
#    make plot         – regenerate plot from existing CSVs
#    make clean        – remove binaries and results
#    make help         – show this message
# ===========================================================

CXX      := g++
NVCC     := nvcc
PYTHON   := python3

CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra -lm
NVCCFLAGS := -std=c++17 -O3 --expt-relaxed-constexpr --use_fast_math

# Auto-detect CUDA arch from nvidia-smi (fallback: 75)
CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap \
                 --format=csv,noheader 2>/dev/null \
                 | head -1 | tr -d '.' 2>/dev/null || echo 75)

CPU_SRC  := cpu/ml_cpu.cpp
CUDA_SRC := cuda/ml_cuda.cu
CPU_BIN  := cpu/ml_cpu
CUDA_BIN := cuda/ml_cuda
RESULTS  := results

# Detect if nvcc is available
HAS_NVCC := $(shell command -v $(NVCC) 2>/dev/null && echo yes || echo no)

.PHONY: all cpu cuda run run_cpu plot clean help

# ── Default: build everything available ──────────────────────
all: cpu
ifeq ($(HAS_NVCC),yes)
all: cuda
endif

# ── CPU binary ───────────────────────────────────────────────
cpu: $(CPU_BIN)

$(CPU_BIN): $(CPU_SRC)
	@echo "[CXX]  Compiling CPU binary..."
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "[CXX]  Done → $@"

# ── CUDA binary ──────────────────────────────────────────────
cuda: $(CUDA_BIN)

$(CUDA_BIN): $(CUDA_SRC)
	@echo "[NVCC] Compiling CUDA binary (sm_$(CUDA_ARCH))..."
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(CUDA_ARCH) -o $@ $<
	@echo "[NVCC] Done → $@"

# ── Full benchmark run ───────────────────────────────────────
run: all
	@mkdir -p $(RESULTS)
	@bash scripts/run_benchmark.sh

# ── CPU-only benchmark ───────────────────────────────────────
run_cpu: cpu
	@mkdir -p $(RESULTS)
	@bash scripts/run_benchmark.sh --skip-cuda

# ── Regenerate plot from existing CSVs ───────────────────────
plot:
	$(PYTHON) scripts/plot_results.py \
		--combined $(RESULTS)/combined_results.csv \
		--speedup  $(RESULTS)/speedup_summary.csv  \
		--output   $(RESULTS)/benchmark_plots.png

# ── Clean ────────────────────────────────────────────────────
clean:
	@rm -f $(CPU_BIN) $(CUDA_BIN)
	@rm -rf $(RESULTS)
	@echo "[clean] Removed binaries and results/"

# ── Help ─────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  ml_benchmark  —  CPU vs CUDA ML algorithm suite"
	@echo ""
	@echo "  make all       Build CPU + CUDA (if nvcc available)"
	@echo "  make cpu       Build CPU binary only"
	@echo "  make cuda      Build CUDA binary only"
	@echo "  make run       Build + run full benchmark + plot"
	@echo "  make run_cpu   Build + run CPU-only + plot"
	@echo "  make plot      Regenerate plot from existing CSVs"
	@echo "  make clean     Remove binaries and results/"
	@echo ""
	@echo "  CUDA arch detected: sm_$(CUDA_ARCH)"
	@echo "  nvcc available:     $(HAS_NVCC)"
	@echo ""
