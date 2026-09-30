// Result type returned by every logistic regression optimizer.
#pragma once

#include <string>

// Timing covers the training loop only. Loss is the average over the last epoch.
struct OptResult {
    double time_ms;
    double final_loss;
    double accuracy;
    int    n_samples;
    std::string tag;
};
