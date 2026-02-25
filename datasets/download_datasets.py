"""
download_datasets.py
Downloads publicly available medical signal datasets from PhysioNet
using the wfdb library.

Datasets:
  - MIT-BIH Arrhythmia Database (mitdb)  → datasets/mit-bih/
  - BIDMC PPG and Respiration Dataset     → datasets/bidmc/

Install dependency first:
  pip install wfdb

Usage:
  python download_datasets.py
"""

import os
import sys

try:
    import wfdb
except ImportError:
    print("[ERROR] wfdb is not installed. Run: pip install wfdb")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── MIT-BIH Arrhythmia Database (all 48 records) ─────────────────────────────
MITBIH_DIR = os.path.join(BASE_DIR, "mit-bih")
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222",
    "223", "228", "230", "231", "232", "233", "234",
]

# ── BIDMC PPG Dataset (all 53 records) ───────────────────────────────────────
BIDMC_DIR = os.path.join(BASE_DIR, "bidmc")
BIDMC_RECORDS = [f"{i:02d}" for i in range(1, 54)]   # bidmc01 – bidmc53


def download_mitbih():
    print("\n── MIT-BIH Arrhythmia Database ──────────────────────────────")
    os.makedirs(MITBIH_DIR, exist_ok=True)
    for rec in MITBIH_RECORDS:
        target = os.path.join(MITBIH_DIR, rec + ".hea")
        if os.path.exists(target):
            print(f"  [SKIP] {rec} already downloaded.")
            continue
        print(f"  Downloading record {rec} ...", end=" ", flush=True)
        try:
            wfdb.dl_database("mitdb", dl_dir=MITBIH_DIR, records=[rec])
            print("OK")
        except Exception as e:
            print(f"FAILED ({e})")
    print(f"  Saved to: {MITBIH_DIR}")


def download_bidmc():
    print("\n── BIDMC PPG and Respiration Dataset ────────────────────────")
    os.makedirs(BIDMC_DIR, exist_ok=True)
    for num in BIDMC_RECORDS:
        rec = f"bidmc{num}"
        target = os.path.join(BIDMC_DIR, rec + ".hea")
        if os.path.exists(target):
            print(f"  [SKIP] {rec} already downloaded.")
            continue
        print(f"  Downloading record {rec} ...", end=" ", flush=True)
        try:
            wfdb.dl_database("bidmc", dl_dir=BIDMC_DIR, records=[rec])
            print("OK")
        except Exception as e:
            print(f"FAILED ({e})")
    print(f"  Saved to: {BIDMC_DIR}")


if __name__ == "__main__":
    print("PhysioNet Dataset Downloader — Spike-Edge Cardiac Project")
    download_mitbih()
    download_bidmc()
    print("\n[DONE] All datasets processed.")
    print("  MIT-BIH ->", MITBIH_DIR)
    print("  BIDMC   ->", BIDMC_DIR)
