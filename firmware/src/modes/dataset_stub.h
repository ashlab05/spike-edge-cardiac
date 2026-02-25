#pragma once
#include "../../config.h"

#ifdef MODE_DATASET

/**
 * dataset_stub.h
 * Dataset replay mode for embedded validation.
 * Reads pre-defined physiological data arrays compiled into flash,
 * allowing the SNN to be validated on the ESP32 without live sensors.
 */

void dataset_stub_init();

/**
 * Read the next sample from the embedded dataset.
 * Returns false when all samples have been consumed.
 */
bool dataset_stub_read(float *hr, float *spo2, float *temp, int *expected_label);

/**
 * Reset the replay cursor to the beginning.
 */
void dataset_stub_reset();

#endif  // MODE_DATASET
