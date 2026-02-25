// lif_neuron.cpp — Leaky Integrate-and-Fire neuron
#include "lif_neuron.h"
#include "../../config.h"

void lif_init(LIFNeuron *n) {
    n->v = 0.0f;
}

int lif_step(LIFNeuron *n, float I) {
    n->v = LIF_ALPHA * n->v + I;
    if (n->v >= LIF_THRESHOLD) {
        n->v = 0.0f;
        return 1;
    }
    return 0;
}
