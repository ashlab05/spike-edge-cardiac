#pragma once

int  encode_spike(float delta, float threshold);
void encode_spikes(float hr, float spo2, float temp,
                   float prev_hr, float prev_spo2, float prev_temp,
                   int spikes[4]);
