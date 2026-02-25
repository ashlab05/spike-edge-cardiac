#pragma once

typedef struct {
    float v;   // membrane potential
} LIFNeuron;

void lif_init(LIFNeuron *n);
int  lif_step(LIFNeuron *n, float I);
