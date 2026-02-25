"""
generate_datasets.py
Generate synthetic multimodal physiological datasets (ECG, PPG, Temperature)
for offline validation of the Spike-Edge Cardiac Anomaly Detection system.

Outputs:
  datasets/synthetic-ecg/   — synthetic ECG waveforms with annotated anomalies
  datasets/synthetic-ppg/   — synthetic PPG waveforms with annotated anomalies
  datasets/combined/        — combined multimodal feature CSVs ready for SNN evaluation

Usage:
  pip install numpy scipy
  python generate_datasets.py
"""

import os
import csv
import math
import random

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directories
ECG_DIR      = os.path.join(BASE_DIR, "synthetic-ecg")
PPG_DIR      = os.path.join(BASE_DIR, "synthetic-ppg")
COMBINED_DIR = os.path.join(BASE_DIR, "combined")

# ── Generation parameters ────────────────────────────────────────────────────
FS_ECG = 360          # MIT-BIH standard sampling rate (Hz)
FS_PPG = 125          # BIDMC standard sampling rate (Hz)
TICK_RATE = 10        # feature extraction rate (Hz), matches firmware

NUM_ECG_RECORDS = 30  # number of synthetic ECG records
NUM_PPG_RECORDS = 30  # number of synthetic PPG records
NUM_COMBINED    = 50  # number of combined multimodal records

RECORD_DURATION_S = 60  # seconds per record


# ── Synthetic ECG generation ─────────────────────────────────────────────────

def _ecg_beat(t, hr_bpm):
    """Generate a single PQRST-like waveform cycle at a given heart rate."""
    period = 60.0 / hr_bpm
    phase = (t % period) / period  # 0..1 within beat

    # Simplified PQRST morphology
    p = 0.15 * np.exp(-((phase - 0.10) ** 2) / (2 * 0.01 ** 2))
    q = -0.10 * np.exp(-((phase - 0.17) ** 2) / (2 * 0.005 ** 2))
    r = 1.00 * np.exp(-((phase - 0.20) ** 2) / (2 * 0.008 ** 2))
    s = -0.20 * np.exp(-((phase - 0.23) ** 2) / (2 * 0.005 ** 2))
    t_wave = 0.25 * np.exp(-((phase - 0.35) ** 2) / (2 * 0.02 ** 2))

    return p + q + r + s + t_wave


def generate_ecg_record(record_id, duration_s=RECORD_DURATION_S):
    """
    Generate a synthetic ECG record with normal and anomaly segments.
    Returns (time_array, ecg_signal, labels_per_sample).
    """
    n_samples = duration_s * FS_ECG
    t = np.arange(n_samples) / FS_ECG

    ecg = np.zeros(n_samples)
    labels = np.zeros(n_samples, dtype=int)

    # Randomly place 2-4 anomaly windows (tachycardia / arrhythmia)
    num_anomalies = random.randint(2, 4)
    anomaly_windows = []
    for _ in range(num_anomalies):
        start = random.uniform(5, duration_s - 10)
        length = random.uniform(3, 8)
        anomaly_windows.append((start, start + length))

    for i in range(n_samples):
        ts = t[i]
        in_anomaly = any(s <= ts < e for s, e in anomaly_windows)

        if in_anomaly:
            hr = random.uniform(110, 150)
            labels[i] = 1
        else:
            hr = random.uniform(60, 80)

        ecg[i] = _ecg_beat(ts, hr)

    # Add realistic noise
    ecg += np.random.normal(0, 0.02, n_samples)
    # Baseline wander
    ecg += 0.1 * np.sin(2 * np.pi * 0.15 * t)

    return t, ecg, labels, anomaly_windows


def save_ecg_record(record_id, t, ecg, labels):
    """Save synthetic ECG as CSV."""
    os.makedirs(ECG_DIR, exist_ok=True)
    path = os.path.join(ECG_DIR, f"ecg_{record_id:03d}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "ecg_mv", "label"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}", f"{ecg[i]:.6f}", labels[i]])
    return path


# ── Synthetic PPG generation ─────────────────────────────────────────────────

def generate_ppg_record(record_id, duration_s=RECORD_DURATION_S):
    """
    Generate a synthetic PPG record with normal and anomaly segments.
    Returns (time_array, ppg_signal, spo2_values, labels_per_sample).
    """
    n_samples = duration_s * FS_PPG
    t = np.arange(n_samples) / FS_PPG

    ppg = np.zeros(n_samples)
    spo2 = np.zeros(n_samples)
    labels = np.zeros(n_samples, dtype=int)

    num_anomalies = random.randint(2, 4)
    anomaly_windows = []
    for _ in range(num_anomalies):
        start = random.uniform(5, duration_s - 10)
        length = random.uniform(3, 8)
        anomaly_windows.append((start, start + length))

    for i in range(n_samples):
        ts = t[i]
        in_anomaly = any(s <= ts < e for s, e in anomaly_windows)

        if in_anomaly:
            hr = random.uniform(110, 150)
            spo2_val = random.uniform(85, 93)
            amplitude = 0.5 + random.uniform(-0.15, 0.15)
            labels[i] = 1
        else:
            hr = random.uniform(60, 80)
            spo2_val = random.uniform(96, 100)
            amplitude = 1.0 + random.uniform(-0.05, 0.05)

        period = 60.0 / hr
        phase = (ts % period) / period
        # PPG waveform: systolic peak + dicrotic notch
        systolic = amplitude * np.exp(-((phase - 0.25) ** 2) / (2 * 0.015 ** 2))
        dicrotic = 0.3 * amplitude * np.exp(-((phase - 0.45) ** 2) / (2 * 0.02 ** 2))
        ppg[i] = systolic + dicrotic
        spo2[i] = spo2_val

    ppg += np.random.normal(0, 0.01, n_samples)
    return t, ppg, spo2, labels, anomaly_windows


def save_ppg_record(record_id, t, ppg, spo2, labels):
    """Save synthetic PPG as CSV."""
    os.makedirs(PPG_DIR, exist_ok=True)
    path = os.path.join(PPG_DIR, f"ppg_{record_id:03d}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "ppg_au", "spo2_pct", "label"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}", f"{ppg[i]:.6f}", f"{spo2[i]:.1f}", labels[i]])
    return path


# ── Combined multimodal feature dataset ──────────────────────────────────────

def generate_combined_record(record_id, duration_s=RECORD_DURATION_S):
    """
    Generate a combined feature-level record at TICK_RATE Hz.
    Columns: time, hr, spo2, temp, label
    This matches the firmware serial output format.
    """
    n_ticks = duration_s * TICK_RATE
    rows = []

    # Define anomaly windows
    num_anomalies = random.randint(2, 4)
    anomaly_windows = []
    for _ in range(num_anomalies):
        start = random.uniform(5, duration_s - 10)
        length = random.uniform(3, 8)
        anomaly_windows.append((start, start + length))

    base_hr = random.uniform(65, 78)
    base_spo2 = random.uniform(96, 99)
    base_temp = random.uniform(36.4, 37.0)

    for i in range(n_ticks):
        ts = i / TICK_RATE
        in_anomaly = any(s <= ts < e for s, e in anomaly_windows)

        if in_anomaly:
            hr = random.uniform(105, 150) + random.gauss(0, 2)
            spo2 = random.uniform(85, 93) + random.gauss(0, 0.5)
            temp = base_temp + random.uniform(0.3, 1.5) + random.gauss(0, 0.05)
            label = 1
        else:
            hr = base_hr + random.gauss(0, 2)
            spo2 = base_spo2 + random.gauss(0, 0.3)
            temp = base_temp + random.gauss(0, 0.05)
            label = 0

        spo2 = max(70.0, min(100.0, spo2))
        rows.append((ts, hr, spo2, temp, label))

    return rows


def save_combined_record(record_id, rows):
    """Save combined features as CSV (firmware-compatible format)."""
    os.makedirs(COMBINED_DIR, exist_ok=True)
    path = os.path.join(COMBINED_DIR, f"combined_{record_id:03d}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time", "HR", "SpO2", "Temp", "Label"])
        for ts, hr, spo2, temp, label in rows:
            w.writerow([f"{ts:.1f}", f"{hr:.1f}", f"{spo2:.1f}", f"{temp:.2f}", label])
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    print("Synthetic Dataset Generator — Spike-Edge Cardiac Project\n")

    # ECG records
    print(f"── Generating {NUM_ECG_RECORDS} synthetic ECG records ──")
    for i in range(NUM_ECG_RECORDS):
        t, ecg, labels, _ = generate_ecg_record(i)
        path = save_ecg_record(i, t, ecg, labels)
        print(f"  [{i+1:2d}/{NUM_ECG_RECORDS}] {os.path.basename(path)}")

    # PPG records
    print(f"\n── Generating {NUM_PPG_RECORDS} synthetic PPG records ──")
    for i in range(NUM_PPG_RECORDS):
        t, ppg, spo2, labels, _ = generate_ppg_record(i)
        path = save_ppg_record(i, t, ppg, spo2, labels)
        print(f"  [{i+1:2d}/{NUM_PPG_RECORDS}] {os.path.basename(path)}")

    # Combined multimodal records
    print(f"\n── Generating {NUM_COMBINED} combined multimodal records ──")
    for i in range(NUM_COMBINED):
        rows = generate_combined_record(i)
        path = save_combined_record(i, rows)
        anomaly_count = sum(1 for r in rows if r[4] == 1)
        total = len(rows)
        print(f"  [{i+1:2d}/{NUM_COMBINED}] {os.path.basename(path)}  "
              f"({anomaly_count}/{total} anomaly ticks)")

    print(f"\n[DONE] Synthetic datasets generated.")
    print(f"  ECG      -> {ECG_DIR}")
    print(f"  PPG      -> {PPG_DIR}")
    print(f"  Combined -> {COMBINED_DIR}")


if __name__ == "__main__":
    main()
