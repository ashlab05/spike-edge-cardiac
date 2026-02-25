"""
spike_encoder.py
Convert feature deltas into binary spike representations.
"""


# Default thresholds — tune based on dataset statistics
DEFAULT_THRESHOLDS = {
    "hr_mean":  8.0,
    "hr_std":   3.0,
    "rr_delta": 0.15,
    "spo2":     2.0,
    "temp":     0.4,
}


def encode_features(feature_deltas: dict, thresholds: dict = None) -> dict:
    """
    Convert a dict of absolute feature changes into binary spikes.
    Returns dict with same keys, values in {0, 1}.
    """
    thr = thresholds or DEFAULT_THRESHOLDS
    spikes = {}
    for key, delta in feature_deltas.items():
        threshold = thr.get(key, 1.0)
        spikes[key] = 1 if abs(delta) > threshold else 0
    return spikes


def encode_vector(feature_deltas: dict, keys: list, thresholds: dict = None) -> list:
    """
    Encode selected features into an ordered spike vector.
    Returns list of ints (0 or 1) in the same order as 'keys'.
    """
    encoded = encode_features(feature_deltas, thresholds)
    return [encoded.get(k, 0) for k in keys]
