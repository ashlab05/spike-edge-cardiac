"""
SimulatorEngine: generates synthetic physiological signals and runs
an inline LIF (Leaky Integrate-and-Fire) SNN for anomaly detection.

Normal baseline:  HR=72, SpO2=98, Temp=36.8
Anomaly burst:    HR=120, SpO2=92, Temp=37.2  (auto-reverts after 5s)
"""

import time
import threading
import random
import math


# ── LIF / SNN parameters ─────────────────────────────────────────────────────
LIF_ALPHA = 0.9          # membrane decay
LIF_THRESHOLD = 1.0      # spike threshold

# spike encoding thresholds (|delta| > threshold → spike=1)
SPIKE_THR_HR   = 8.0     # bpm
SPIKE_THR_SPO2 = 2.0     # %
SPIKE_THR_TEMP = 0.4     # °C

# hidden-layer weights [HR, SpO2, Temp, RR_proxy] → hidden[0..4]
W_INPUT_HIDDEN = [
    [0.50, 0.40, 0.20, 0.30],   # hidden 0
    [0.45, 0.35, 0.25, 0.20],   # hidden 1
    [0.40, 0.50, 0.15, 0.35],   # hidden 2
    [0.35, 0.30, 0.30, 0.40],   # hidden 3
    [0.55, 0.45, 0.10, 0.25],   # hidden 4
]
W_HIDDEN_OUTPUT = [0.4, 0.35, 0.3, 0.25, 0.45]   # hidden → output

# ── Normal baseline ───────────────────────────────────────────────────────────
BASELINE = {"hr": 72.0, "spo2": 98.0, "temp": 36.8}
ANOMALY_VALS = {"hr": 120.0, "spo2": 92.0, "temp": 37.2}
NOISE = {"hr": 2.0, "spo2": 0.5, "temp": 0.1}


class SimulatorEngine:
    def __init__(self):
        self._lock = threading.Lock()

        # current "target" values (may be overridden by UI sliders)
        self._target = dict(BASELINE)
        self._custom_mode = False          # True = slider overrides active

        # anomaly injection state
        self._inject_active = False
        self._inject_end_time = 0.0

        # previous tick values for delta computation
        self._prev = dict(BASELINE)

        # LIF membrane potentials: 5 hidden + 1 output
        self._v_hidden = [0.0] * 5
        self._v_output = 0.0

        # published state (read by WebSocket broadcaster)
        self._state = {
            "time": 0.0,
            "hr": BASELINE["hr"],
            "spo2": BASELINE["spo2"],
            "temp": BASELINE["temp"],
            "anomaly": 0,
            "mode": "auto",
        }
        self._start_time = time.time()

    # ── public API ────────────────────────────────────────────────────────────

    def set_config(self, hr: float, spo2: float, temp: float):
        """Called by REST /simulator/config (slider update)."""
        with self._lock:
            self._target = {"hr": float(hr), "spo2": float(spo2), "temp": float(temp)}
            self._custom_mode = True
            self._state["mode"] = "custom"

    def inject_anomaly(self, duration: float = 5.0):
        """Trigger a timed anomaly burst."""
        with self._lock:
            self._inject_active = True
            self._inject_end_time = time.time() + duration

    def reset(self):
        """Return to auto baseline mode."""
        with self._lock:
            self._target = dict(BASELINE)
            self._custom_mode = False
            self._inject_active = False
            self._state["mode"] = "auto"

    def get_state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def tick(self):
        """Advance simulation by one timestep. Call at ~10 Hz."""
        now = time.time()
        elapsed = now - self._start_time

        with self._lock:
            # ── resolve target values ─────────────────────────────────────
            if self._inject_active:
                if now < self._inject_end_time:
                    target = dict(ANOMALY_VALS)
                else:
                    self._inject_active = False
                    target = dict(self._target)
            else:
                target = dict(self._target)

            # add Gaussian noise
            hr   = target["hr"]   + random.gauss(0, NOISE["hr"])
            spo2 = target["spo2"] + random.gauss(0, NOISE["spo2"])
            temp = target["temp"] + random.gauss(0, NOISE["temp"])

            # clamp to physiological ranges
            hr   = max(30.0, min(220.0, hr))
            spo2 = max(70.0, min(100.0, spo2))
            temp = max(34.0, min(42.0, temp))

            # ── spike encoding (delta-based) ──────────────────────────────
            spikes = [
                1 if abs(hr   - self._prev["hr"])   > SPIKE_THR_HR   else 0,
                1 if abs(spo2 - self._prev["spo2"]) > SPIKE_THR_SPO2 else 0,
                1 if abs(temp - self._prev["temp"]) > SPIKE_THR_TEMP else 0,
                # RR proxy: high-frequency HR variation → 4th spike channel
                1 if abs(hr - self._prev["hr"]) > 4.0 else 0,
            ]

            # ── LIF forward pass: input → hidden ─────────────────────────
            h_spikes = []
            for i in range(5):
                I = sum(W_INPUT_HIDDEN[i][j] * spikes[j] for j in range(4))
                self._v_hidden[i] = LIF_ALPHA * self._v_hidden[i] + I
                if self._v_hidden[i] >= LIF_THRESHOLD:
                    h_spikes.append(1)
                    self._v_hidden[i] = 0.0   # reset
                else:
                    h_spikes.append(0)

            # ── LIF forward pass: hidden → output ─────────────────────────
            I_out = sum(W_HIDDEN_OUTPUT[i] * h_spikes[i] for i in range(5))
            self._v_output = LIF_ALPHA * self._v_output + I_out
            if self._v_output >= LIF_THRESHOLD:
                anomaly = 1
                self._v_output = 0.0
            else:
                anomaly = 0

            # ── update state ──────────────────────────────────────────────
            self._prev = {"hr": hr, "spo2": spo2, "temp": temp}
            self._state.update({
                "time": round(elapsed, 2),
                "hr":   round(hr, 1),
                "spo2": round(spo2, 1),
                "temp": round(temp, 2),
                "anomaly": anomaly,
            })
