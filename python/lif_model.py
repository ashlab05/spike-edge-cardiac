"""
lif_model.py
Leaky Integrate-and-Fire (LIF) neuron and SNN network for anomaly detection.

Architecture: 4 inputs → 5 hidden LIF neurons → 1 output LIF neuron
"""

import numpy as np


class LIFNeuron:
    """Single Leaky Integrate-and-Fire neuron."""

    def __init__(self, alpha: float = 0.9, threshold: float = 1.0):
        self.alpha     = alpha      # membrane decay constant
        self.threshold = threshold
        self.v         = 0.0        # membrane potential

    def step(self, I: float) -> int:
        """
        Advance one timestep.
        I: input current (weighted sum of pre-synaptic spikes).
        Returns 1 (spike) or 0 (no spike).
        """
        self.v = self.alpha * self.v + I
        if self.v >= self.threshold:
            self.v = 0.0
            return 1
        return 0

    def reset(self):
        self.v = 0.0


class SNNNetwork:
    """
    Fully-connected 3-layer SNN: input → hidden → output.
    """

    def __init__(
        self,
        n_input:  int = 4,
        n_hidden: int = 5,
        n_output: int = 1,
        alpha:    float = 0.9,
        threshold: float = 1.0,
    ):
        self.n_input  = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Xavier-style initialisation
        self.W_ih = np.random.uniform(
            0.2, 0.6, size=(n_hidden, n_input)
        )
        self.W_ho = np.random.uniform(
            0.2, 0.5, size=(n_output, n_hidden)
        )

        self.hidden = [LIFNeuron(alpha, threshold) for _ in range(n_hidden)]
        self.output = [LIFNeuron(alpha, threshold) for _ in range(n_output)]

    def forward(self, spikes_in: list) -> list:
        """
        Forward pass for one timestep.
        spikes_in: list of ints (0/1), length = n_input.
        Returns list of output spikes, length = n_output.
        """
        x = np.array(spikes_in, dtype=float)

        # hidden layer
        h_spikes = []
        for i, neuron in enumerate(self.hidden):
            I = float(self.W_ih[i] @ x)
            h_spikes.append(neuron.step(I))

        h = np.array(h_spikes, dtype=float)

        # output layer
        out_spikes = []
        for i, neuron in enumerate(self.output):
            I = float(self.W_ho[i] @ h)
            out_spikes.append(neuron.step(I))

        return out_spikes

    def reset_state(self):
        for n in self.hidden: n.reset()
        for n in self.output:  n.reset()

    def set_weights(self, W_ih: np.ndarray, W_ho: np.ndarray):
        assert W_ih.shape == (self.n_hidden, self.n_input)
        assert W_ho.shape == (self.n_output, self.n_hidden)
        self.W_ih = W_ih.copy()
        self.W_ho = W_ho.copy()
