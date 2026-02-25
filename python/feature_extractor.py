"""
feature_extractor.py
Extract physiological features from raw ECG and PPG signals.
"""

import numpy as np
import wfdb.processing as wp


def extract_ecg_features(ecg_signal: np.ndarray, fs: int) -> dict:
    """
    Extract HR and RR-interval features from an ECG signal segment.
    Returns dict: {hr_mean, hr_std, rr_mean, rr_std, rr_delta}
    """
    # R-peak detection
    r_peaks = wp.gqrs_detect(ecg_signal, fs=fs)
    if len(r_peaks) < 2:
        return {"hr_mean": 0, "hr_std": 0, "rr_mean": 0, "rr_std": 0, "rr_delta": 0}

    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    hr_values    = 60.0 / rr_intervals

    return {
        "hr_mean":  float(np.mean(hr_values)),
        "hr_std":   float(np.std(hr_values)),
        "rr_mean":  float(np.mean(rr_intervals)),
        "rr_std":   float(np.std(rr_intervals)),
        "rr_delta": float(np.max(rr_intervals) - np.min(rr_intervals)),
    }


def extract_ppg_features(ppg_signal: np.ndarray, fs: int) -> dict:
    """
    Extract SpO2 proxy and amplitude variability from a PPG signal segment.
    Returns dict: {spo2_proxy, amplitude_mean, amplitude_std}
    """
    normalized = (ppg_signal - ppg_signal.min()) / (ppg_signal.ptp() + 1e-9)
    # Rough perfusion index as SpO2 proxy (not a calibrated measurement)
    ac = ppg_signal.ptp()
    dc = np.abs(ppg_signal.mean())
    spo2_proxy = 100.0 - (ac / (dc + 1e-9)) * 5   # heuristic mapping

    return {
        "spo2_proxy":     float(np.clip(spo2_proxy, 70, 100)),
        "amplitude_mean": float(normalized.mean()),
        "amplitude_std":  float(normalized.std()),
    }


def compute_deltas(current: dict, previous: dict) -> dict:
    """Compute absolute differences between two feature dicts."""
    return {k: abs(current[k] - previous.get(k, current[k])) for k in current}
