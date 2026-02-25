"""
baseline_threshold.py
Classical threshold-based anomaly detection (baseline for comparison).
"""


# Reference thresholds (configurable)
THRESHOLD_HR_HIGH   = 100.0   # bpm
THRESHOLD_SPO2_LOW  = 94.0    # %
THRESHOLD_TEMP_HIGH = 38.0    # °C


def detect(hr: float, spo2: float, temp: float) -> int:
    """
    Returns 1 (anomaly) if any threshold is breached, else 0.
    """
    if hr > THRESHOLD_HR_HIGH:
        return 1
    if spo2 < THRESHOLD_SPO2_LOW:
        return 1
    if temp > THRESHOLD_TEMP_HIGH:
        return 1
    return 0


def detect_batch(records: list) -> list:
    """
    records: list of dicts with keys 'hr', 'spo2', 'temp'.
    Returns list of predictions (0 or 1).
    """
    return [detect(r["hr"], r["spo2"], r["temp"]) for r in records]
