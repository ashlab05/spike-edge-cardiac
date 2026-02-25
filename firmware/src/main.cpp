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
#elif defined(MODE_DATASET)
  #include "modes/dataset_stub.h"
#endif

// ── Previous-tick values for delta computation ────────────────────────────────
static float prev_hr   = SIM_NORMAL_HR;
static float prev_spo2 = SIM_NORMAL_SPO2;
static float prev_temp = 36.8f;

static unsigned long last_tick = 0;

#ifdef MODE_DATASET
static int ds_correct = 0;
static int ds_total   = 0;
static bool ds_done   = false;
#endif

void setup() {
    logger_init();

#ifdef MODE_SIMULATION
    simulated_signals_init();
#elif defined(MODE_HARDWARE)
    hardware_sensors_init();
#elif defined(MODE_DATASET)
    dataset_stub_init();
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
#elif defined(MODE_DATASET)
    if (ds_done) return;
    int expected_label = 0;
    if (!dataset_stub_read(&hr, &spo2, &temp, &expected_label)) {
        // Dataset exhausted — print summary
        Serial.println("──────────────────────────────────");
        Serial.print("[DATASET] Finished. Accuracy: ");
        Serial.print(ds_total > 0 ? (100.0f * ds_correct / ds_total) : 0.0f, 1);
        Serial.print("% (");
        Serial.print(ds_correct);
        Serial.print("/");
        Serial.print(ds_total);
        Serial.println(")");
        ds_done = true;
        return;
    }
#endif

    // ── Spike encoding ────────────────────────────────────────────────────────
    int spikes[4];
    encode_spikes(hr, spo2, temp, prev_hr, prev_spo2, prev_temp, spikes);

    // ── SNN inference ─────────────────────────────────────────────────────────
    int anomaly = snn_forward(spikes);

    // ── Serial log ────────────────────────────────────────────────────────────
    logger_print((float)now / 1000.0f, hr, spo2, temp, anomaly);

#ifdef MODE_DATASET
    // Track accuracy against ground truth
    if (anomaly == expected_label) ds_correct++;
    ds_total++;
#endif

    // ── Update previous values ────────────────────────────────────────────────
    prev_hr   = hr;
    prev_spo2 = spo2;
    prev_temp = temp;
}
