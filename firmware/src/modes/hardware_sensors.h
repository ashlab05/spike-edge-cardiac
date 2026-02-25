#pragma once
#include "../../config.h"

#ifdef MODE_HARDWARE
void hardware_sensors_init();
void hardware_sensors_read(float *hr, float *spo2, float *temp);
#endif
