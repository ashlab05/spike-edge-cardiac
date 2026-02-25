// feature_extractor.cpp — Temporal feature extraction for spike encoding
// Computes per-tick deltas and exponential-moving-average rate of change.

#include "feature_extractor.h"
#include "../../config.h"

// EMA smoothing factor (0 = no smoothing, 1 = instant).
// 0.3 gives a gentle 3-tick smoothing window at 10 Hz.
#define EMA_ALPHA 0.3f

static float ema_hr_rate;
static float ema_temp_rate;

static float fabsf_local(float x) { return x < 0 ? -x : x; }

void feature_extractor_init() {
    ema_hr_rate   = 0.0f;
    ema_temp_rate = 0.0f;
}

void feature_extract(float hr, float spo2, float temp,
                     float prev_hr, float prev_spo2, float prev_temp,
                     Features *out) {
    // Raw absolute deltas
    out->hr_delta   = fabsf_local(hr   - prev_hr);
    out->spo2_delta = fabsf_local(spo2 - prev_spo2);
    out->temp_delta = fabsf_local(temp - prev_temp);

    // Smoothed rate-of-change via EMA
    ema_hr_rate   = EMA_ALPHA * out->hr_delta   + (1.0f - EMA_ALPHA) * ema_hr_rate;
    ema_temp_rate = EMA_ALPHA * out->temp_delta  + (1.0f - EMA_ALPHA) * ema_temp_rate;

    out->hr_rate   = ema_hr_rate;
    out->temp_rate = ema_temp_rate;
}
