// CSV output for benchmark results.
// main opens G_CSV and every algorithm result is appended with write_row.
#pragma once

#include <fstream>
#include <string>

inline std::ofstream G_CSV;

// One row per algorithm, size and metric. The last column marks the device.
static void write_row(const std::string& algo, int n, int feat,
                      double time_ms, const std::string& mname, double mval) {
    G_CSV << algo << "," << n << "," << feat << "," << time_ms
          << "," << mname << "," << mval << ",CPU\n";
}
