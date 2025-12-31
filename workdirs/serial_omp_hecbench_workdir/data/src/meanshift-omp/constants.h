#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

namespace mean_shift::gpu {

    // Hyperparameters
    constexpr float RADIUS = 60;
    constexpr float SIGMA = 4;
    constexpr float DBL_SIGMA_SQ = (2 * SIGMA * SIGMA);
    constexpr float MIN_DISTANCE = 60;
    constexpr size_t NUM_ITER = 50;
    constexpr float DIST_TO_REAL = 10;
    // Dataset
    constexpr int N = 5000;
    constexpr int D = 3;
    constexpr int M = 3;
    // Device
    // THREADS is the per-team thread limit used as a hint for the
    // OpenMP offload runtime. Keep a reasonable thread limit so that
    // each team has enough threads to make progress.
    constexpr int THREADS = 64;
    // Increase the number of teams launched on the device to improve
    // occupancy. This value is a launch-hint only and does not change
    // the problem geometry or iteration counts. Using one team per
    // output point gives the device more blocks to schedule across SMs
    // which helps reduce low-wave occupancy reported by the profiler.
    constexpr int BLOCKS = N;
    constexpr int TILE_WIDTH = THREADS;

}

#endif
