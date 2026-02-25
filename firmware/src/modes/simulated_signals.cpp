#include "../../config.h"
#include "simulated_signals.h"
#include <Arduino.h>
#include <math.h>

#ifdef MODE_SIMULATION

static float _hr   = SIM_NORMAL_HR;
static float _spo2 = SIM_NORMAL_SPO2;
static float _temp = 36.8f;

static float randf(float low, float high) {
    return low + ((float)random(0, 1000) / 1000.0f) * (high - low);
}

void simulated_signals_init() {
    randomSeed(analogRead(0));
}

void simulated_signals_read(float *hr, float *spo2, float *temp) {
    unsigned long t = millis();

    if (t >= SIM_ANOMALY_START && t < SIM_ANOMALY_END) {
        // Anomaly window
        _hr   = SIM_ANOMALY_HR   + randf(-2.0f, 2.0f);
        _spo2 = SIM_ANOMALY_SPO2 + randf(-0.5f, 0.5f);
        _temp = 37.2f             + randf(-0.1f, 0.1f);
    } else {
        // Normal baseline
        _hr   = SIM_NORMAL_HR   + randf(-2.0f, 2.0f);
        _spo2 = SIM_NORMAL_SPO2 + randf(-0.5f, 0.5f);
        _temp = 36.8f            + randf(-0.1f, 0.1f);
    }

    *hr   = _hr;
    *spo2 = _spo2;
    *temp = _temp;
}

#endif  // MODE_SIMULATION
