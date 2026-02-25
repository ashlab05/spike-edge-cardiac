#pragma once
#include "../../config.h"

#ifdef MODE_SIMULATION
void simulated_signals_init();
void simulated_signals_read(float *hr, float *spo2, float *temp);
#endif
