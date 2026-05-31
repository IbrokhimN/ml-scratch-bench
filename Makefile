CXX       := g++
NVCC      := nvcc
PYTHON    := python3

CXXFLAGS  := -std=c++17 -O3 -march=native -Wall -lm
NVCCFLAGS := -std=c++17 -O3 --expt-relaxed-constexpr --use_fast_math

CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap \
                 --format=csv,noheader 2>/dev/null \
                 | head -1 | tr -d '.' 2>/dev/null || echo 75)

CPU_SRC  := cpu/ml_cpu.cpp
CUDA_SRC := cuda/ml_cuda.cu
CPU_BIN  := cpu/ml_cpu
CUDA_BIN := cuda/ml_cuda
RESULTS  := results

HAS_NVCC := $(shell command -v $(NVCC) 2>/dev/null && echo yes || echo no)

.PHONY: all cpu cuda run run_cpu plot clean help

all: cpu
ifeq ($(HAS_NVCC),yes)
all: cuda
endif

cpu: $(CPU_BIN)
$(CPU_BIN): $(CPU_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

cuda: $(CUDA_BIN)
$(CUDA_BIN): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) -arch=sm_$(CUDA_ARCH) -o $@ $<

run: all
	@mkdir -p $(RESULTS)
	@bash scripts/run_benchmark.sh

run_cpu: cpu
	@mkdir -p $(RESULTS)
	@bash scripts/run_benchmark.sh --skip-cuda

plot:
	$(PYTHON) scripts/plot_results.py \
		--combined $(RESULTS)/combined_results.csv \
		--speedup  $(RESULTS)/speedup_summary.csv  \
		--output   $(RESULTS)/benchmark_plots.png

clean:
	@rm -f $(CPU_BIN) $(CUDA_BIN)
	@rm -rf $(RESULTS)

help:
	@echo ""
	@echo "  make all       build CPU + CUDA (if nvcc found)"
	@echo "  make cpu       build CPU only"
	@echo "  make cuda      build CUDA only"
	@echo "  make run       build + run + plot"
	@echo "  make run_cpu   build + run CPU only + plot"
	@echo "  make plot      replot from existing CSVs"
	@echo "  make clean     remove binaries and results/"
	@echo ""
	@echo "  CUDA arch: sm_$(CUDA_ARCH)   nvcc: $(HAS_NVCC)"
	@echo ""
