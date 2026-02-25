#pragma once

/**
 * feature_extractor.h
 * Temporal feature extraction from raw physiological signals.
 * Computes smoothed deltas and rate-of-change metrics used upstream
 * by the spike encoder.
 */

typedef struct {
    float hr_delta;      // |HR(t) - HR(t-1)|
    float spo2_delta;    // |SpO2(t) - SpO2(t-1)|
    float temp_delta;    // |Temp(t) - Temp(t-1)|
    float hr_rate;       // smoothed HR rate-of-change (EMA)
    float temp_rate;     // smoothed Temp rate-of-change (EMA)
} Features;

/**
 * Initialise the feature extractor (resets internal EMA state).
 */
void feature_extractor_init();

/**
 * Extract features from the current and previous signal readings.
 * Populates the Features struct with deltas and smoothed rates.
 */
void feature_extract(float hr, float spo2, float temp,
                     float prev_hr, float prev_spo2, float prev_temp,
                     Features *out);
