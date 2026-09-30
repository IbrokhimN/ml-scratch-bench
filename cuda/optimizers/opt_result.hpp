// Result type returned by every GPU logistic regression optimizer.
#pragma once

#include <string>

// Timing covers the training loop only.
struct OptResult { double time_ms, final_loss, accuracy; int n_samples; std::string tag; };
