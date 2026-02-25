// snn_network.cpp — 4-input → 5-hidden → 1-output LIF network
#include "snn_network.h"
#include "lif_neuron.h"
#include "../../config.h"

static LIFNeuron hidden[5];
static LIFNeuron output_neuron;

void snn_init() {
    for (int i = 0; i < 5; i++) lif_init(&hidden[i]);
    lif_init(&output_neuron);
}

int snn_forward(int spikes_in[4]) {
    int h_spikes[5] = {0};

    // Input → Hidden
    for (int i = 0; i < 5; i++) {
        float I = 0.0f;
        for (int j = 0; j < 4; j++) {
            I += W_INPUT_HIDDEN[i][j] * (float)spikes_in[j];
        }
        h_spikes[i] = lif_step(&hidden[i], I);
    }

    // Hidden → Output
    float I_out = 0.0f;
    for (int i = 0; i < 5; i++) {
        I_out += W_HIDDEN_OUTPUT[i] * (float)h_spikes[i];
    }
    return lif_step(&output_neuron, I_out);
}
