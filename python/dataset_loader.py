"""
dataset_loader.py
Load records from MIT-BIH Arrhythmia and BIDMC PPG datasets.
"""

import os
import wfdb
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MITBIH_DIR = os.path.join(BASE, "datasets", "mit-bih")
BIDMC_DIR  = os.path.join(BASE, "datasets", "bidmc")


def load_mitbih(record_id: str):
    """
    Load a MIT-BIH record. Returns (signal_array, annotation, fs).
    signal_array shape: (n_samples, n_channels)
    """
    path = os.path.join(MITBIH_DIR, str(record_id))
    record = wfdb.rdrecord(path)
    ann    = wfdb.rdann(path, "atr")
    return record.p_signal, ann, record.fs


def load_bidmc(record_id: str):
    """
    Load a BIDMC record. Returns (signal_dict, fs).
    signal_dict keys: 'PPG', 'RESP', etc. (channel names)
    """
    path   = os.path.join(BIDMC_DIR, f"bidmc{record_id:>02s}")
    record = wfdb.rdrecord(path)
    signals = {
        record.sig_name[i]: record.p_signal[:, i]
        for i in range(record.n_sig)
    }
    return signals, record.fs


if __name__ == "__main__":
    try:
        sig, ann, fs = load_mitbih("100")
        print(f"MIT-BIH 100: {sig.shape}, fs={fs}, annotations={len(ann.sample)}")
    except FileNotFoundError:
        print("MIT-BIH records not found. Run datasets/download_datasets.py first.")
