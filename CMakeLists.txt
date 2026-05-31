cmake_minimum_required(VERSION 3.18)
project(ml_benchmark LANGUAGES CXX)

# cpp standart
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cpu bin
add_executable(ml_cpu cpu/ml_cpu.cpp)
target_compile_options(ml_cpu PRIVATE -O3 -march=native -Wall)
target_link_libraries(ml_cpu PRIVATE m)

# cuda bin
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)  

    add_executable(ml_cuda cuda/ml_cuda.cu)
    target_compile_options(ml_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-relaxed-constexpr
        --use_fast_math
        -lineinfo
    >)
    target_link_libraries(ml_cuda PRIVATE m)
    message(STATUS "[ml_benchmark] CUDA found → ml_cuda target enabled")
else()
    message(WARNING "[ml_benchmark] CUDA not found → only ml_cpu will be built")
endif()

# targets installation
install(TARGETS ml_cpu DESTINATION bin)
if(TARGET ml_cuda)
    install(TARGETS ml_cuda DESTINATION bin)
endif()
