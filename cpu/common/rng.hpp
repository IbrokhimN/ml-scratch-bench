// Global random number generator shared by every algorithm.
// The fixed seed keeps datasets and results reproducible between runs.
#pragma once

#include <random>

inline std::mt19937_64 G_RNG(0xDEADBEEFCAFE1234ULL);

// Uniform float in the range 0 to 1.
inline float randf()  { return std::uniform_real_distribution<float>(0.f, 1.f)(G_RNG); }

// Standard normal float.
inline float randnf() { return std::normal_distribution<float>(0.f, 1.f)(G_RNG); }

// Uniform integer in the range lo to hi minus one.
inline int   randi(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi-1)(G_RNG); }
