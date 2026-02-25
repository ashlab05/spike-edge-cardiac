// main.cpp — Spike-Edge Cardiac Anomaly Detection Firmware
// Target: ESP32-S3 DevKitC-1 via PlatformIO / Arduino framework

#include <Arduino.h>
#include "config.h"
#include "processing/spike_encoder.h"
#include "snn/snn_network.h"
#include "output/logger.h"

#ifdef MODE_SIMULATION
  #include "modes/simulated_signals.h"
#elif defined(MODE_HARDWARE)
  #include "modes/hardware_sensors.h"
#endif

// ── Previous-tick values for delta computation ────────────────────────────────
static float prev_hr   = SIM_NORMAL_HR;
static float prev_spo2 = SIM_NORMAL_SPO2;
static float prev_temp = 36.8f;

static unsigned long last_tick = 0;

void setup() {
    logger_init();

#ifdef MODE_SIMULATION
    simulated_signals_init();
#elif defined(MODE_HARDWARE)
    hardware_sensors_init();
#endif

    snn_init();
    last_tick = millis();
}

void loop() {
    unsigned long now = millis();
    if (now - last_tick < TICK_INTERVAL_MS) return;
    last_tick = now;

    float hr = 0, spo2 = 0, temp = 0;

    // ── Read signals ──────────────────────────────────────────────────────────
#ifdef MODE_SIMULATION
    simulated_signals_read(&hr, &spo2, &temp);
#elif defined(MODE_HARDWARE)
    hardware_sensors_read(&hr, &spo2, &temp);
#endif

    // ── Spike encoding ────────────────────────────────────────────────────────
    int spikes[4];
    encode_spikes(hr, spo2, temp, prev_hr, prev_spo2, prev_temp, spikes);

    // ── SNN inference ─────────────────────────────────────────────────────────
    int anomaly = snn_forward(spikes);

    // ── Serial log ────────────────────────────────────────────────────────────
    logger_print((float)now / 1000.0f, hr, spo2, temp, anomaly);

    // ── Update previous values ────────────────────────────────────────────────
    prev_hr   = hr;
    prev_spo2 = spo2;
    prev_temp = temp;
}
