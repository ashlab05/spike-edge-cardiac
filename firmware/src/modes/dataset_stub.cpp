// dataset_stub.cpp — Embedded dataset replay for SNN validation
// Feeds a small compiled-in dataset through the same pipeline used
// by simulation and hardware modes, allowing accuracy measurement
// directly on the ESP32-S3.

#include "../../config.h"
#include "dataset_stub.h"
#include <Arduino.h>

#ifdef MODE_DATASET

// ── Compiled-in test dataset ────────────────────────────────────────────────
// Each row: {HR, SpO2, Temp, expected_label}
// This data mirrors the combined CSV format produced by generate_datasets.py.
// Replace or extend with longer sequences as needed.

typedef struct { float hr; float spo2; float temp; int label; } DataRow;

static const DataRow DATASET[] = {
    // Normal baseline (60-80 bpm, 96-100% SpO2, ~36.8 C)
    { 72.1f, 98.2f, 36.80f, 0},
    { 71.8f, 98.0f, 36.81f, 0},
    { 73.0f, 97.9f, 36.79f, 0},
    { 72.5f, 98.1f, 36.82f, 0},
    { 71.6f, 98.3f, 36.78f, 0},
    { 72.9f, 97.8f, 36.80f, 0},
    { 73.2f, 98.0f, 36.83f, 0},
    { 71.4f, 98.2f, 36.81f, 0},
    { 72.7f, 97.9f, 36.79f, 0},
    { 72.0f, 98.1f, 36.80f, 0},
    // Transition into anomaly (sudden HR jump, SpO2 drop)
    { 85.0f, 96.5f, 36.90f, 0},
    { 98.0f, 95.0f, 37.00f, 0},
    // Anomaly window (tachycardia + hypoxemia)
    {120.3f, 92.1f, 37.20f, 1},
    {118.7f, 91.8f, 37.25f, 1},
    {122.0f, 91.5f, 37.30f, 1},
    {119.5f, 92.0f, 37.22f, 1},
    {121.8f, 91.2f, 37.28f, 1},
    {123.1f, 90.8f, 37.35f, 1},
    {120.0f, 91.5f, 37.18f, 1},
    {117.5f, 92.3f, 37.15f, 1},
    {115.2f, 92.8f, 37.10f, 1},
    {110.0f, 93.5f, 37.05f, 1},
    // Recovery to normal
    { 95.0f, 95.0f, 36.95f, 0},
    { 82.0f, 96.8f, 36.88f, 0},
    { 74.5f, 97.5f, 36.82f, 0},
    { 72.8f, 98.0f, 36.80f, 0},
    { 73.1f, 98.1f, 36.79f, 0},
    { 72.0f, 97.9f, 36.81f, 0},
    { 71.5f, 98.2f, 36.80f, 0},
    { 72.3f, 98.0f, 36.78f, 0},
    // Second anomaly burst (bradycardia-like + hypothermia)
    { 55.0f, 97.0f, 36.20f, 1},
    { 52.0f, 96.5f, 36.10f, 1},
    { 48.0f, 96.0f, 36.00f, 1},
    { 50.5f, 96.2f, 36.05f, 1},
    { 53.0f, 96.8f, 36.15f, 1},
    // Return to normal
    { 62.0f, 97.2f, 36.50f, 0},
    { 68.0f, 97.8f, 36.70f, 0},
    { 72.0f, 98.0f, 36.80f, 0},
    { 71.8f, 98.1f, 36.81f, 0},
    { 72.5f, 97.9f, 36.79f, 0},
};

static const int DATASET_LEN = sizeof(DATASET) / sizeof(DATASET[0]);
static int cursor = 0;

void dataset_stub_init() {
    cursor = 0;
    Serial.print("[DATASET] Loaded ");
    Serial.print(DATASET_LEN);
    Serial.println(" samples for replay.");
}

bool dataset_stub_read(float *hr, float *spo2, float *temp, int *expected_label) {
    if (cursor >= DATASET_LEN) return false;

    const DataRow *row = &DATASET[cursor];
    *hr    = row->hr;
    *spo2  = row->spo2;
    *temp  = row->temp;
    *expected_label = row->label;

    cursor++;
    return true;
}

void dataset_stub_reset() {
    cursor = 0;
}

#endif  // MODE_DATASET
