// spike_encoder.cpp
#include "spike_encoder.h"
#include "../../config.h"

int encode_spike(float delta, float threshold) {
    return (delta < 0 ? -delta : delta) > threshold ? 1 : 0;
}

void encode_spikes(float hr, float spo2, float temp,
                   float prev_hr, float prev_spo2, float prev_temp,
                   int spikes[4]) {
    spikes[0] = encode_spike(hr   - prev_hr,   SPIKE_THR_HR);
    spikes[1] = encode_spike(spo2 - prev_spo2, SPIKE_THR_SPO2);
    spikes[2] = encode_spike(temp - prev_temp, SPIKE_THR_TEMP);
    // RR proxy: smaller delta threshold for 4th channel
    spikes[3] = encode_spike(hr   - prev_hr,   4.0f);
}
