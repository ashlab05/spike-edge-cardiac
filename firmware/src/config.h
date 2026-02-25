#pragma once

// ── Operating mode ────────────────────────────────────────────────────────────
// Uncomment exactly one:
#define MODE_SIMULATION
// #define MODE_HARDWARE

// ── Pin assignments ───────────────────────────────────────────────────────────
#define PIN_TEMP_SENSOR   5    // DS18B20 one-wire bus
#define PIN_ECG_ANALOG    4    // AD8232 output (ADC1_CH3)
#define PIN_PPG_SDA       8    // MAX30100 I2C SDA
#define PIN_PPG_SCL       9    // MAX30100 I2C SCL

// ── Simulation parameters ─────────────────────────────────────────────────────
#define SIM_NORMAL_HR      72.0f
#define SIM_NORMAL_SPO2    98.0f
#define SIM_ANOMALY_HR    120.0f
#define SIM_ANOMALY_SPO2   92.0f
#define SIM_ANOMALY_START  10000   // ms from boot
#define SIM_ANOMALY_END    15000   // ms from boot

// ── LIF / SNN parameters ─────────────────────────────────────────────────────
#define LIF_ALPHA          0.9f
#define LIF_THRESHOLD      1.0f

// Spike encoding thresholds
#define SPIKE_THR_HR       8.0f
#define SPIKE_THR_SPO2     2.0f
#define SPIKE_THR_TEMP     0.4f

// Input → Hidden weights (4 inputs × 5 hidden neurons)
static const float W_INPUT_HIDDEN[5][4] = {
    {0.50f, 0.40f, 0.20f, 0.30f},
    {0.45f, 0.35f, 0.25f, 0.20f},
    {0.40f, 0.50f, 0.15f, 0.35f},
    {0.35f, 0.30f, 0.30f, 0.40f},
    {0.55f, 0.45f, 0.10f, 0.25f},
};

// Hidden → Output weights (5 hidden → 1 output)
static const float W_HIDDEN_OUTPUT[5] = {0.40f, 0.35f, 0.30f, 0.25f, 0.45f};

// ── Serial output ─────────────────────────────────────────────────────────────
#define SERIAL_BAUD       115200
#define TICK_INTERVAL_MS  100     // 10 Hz
