// Wall clock aliases used to time every algorithm.
#pragma once

#include <chrono>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;
