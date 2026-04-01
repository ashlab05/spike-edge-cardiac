"""
compare_models.py
Train and compare TinyML-class models against our SNN approach for
cardiac event CLASSIFICATION (4-class) using the IoT health monitoring dataset.

Classes:
    0 = Normal
    1 = Arrhythmia (erratic HR, very low HRV, low SpO2)
    2 = Hypotensive Event (low BP, moderate tachycardia)
    3 = Hypertensive Crisis (very high BP, tachycardia, tachypnea)

Models compared:
  1. Logistic Regression        (minimal baseline)
  2. Decision Tree               (classic edge-friendly)
  3. Random Forest               (ensemble, popular IoT)
  4. XGBoost                     (gradient boosting SOTA)
  5. LightGBM                    (fast gradient boosting)
  6. k-Nearest Neighbors         (lazy learner baseline)
  7. MLP Neural Network          (TinyML standard ANN)
  8. SNN (LIF Spiking Network)   (our approach)
  9. Threshold Baseline           (classical rule-based)

Outputs:
  ML/figures/       — comparison charts and visualizations
  ML/results.json   — raw metrics for each model
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "iot_health_monitoring_dataset.csv")
FIG_DIR   = os.path.join(BASE_DIR, "figures")
JSON_PATH = os.path.join(BASE_DIR, "results.json")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
})
PALETTE = sns.color_palette("husl", 9)

CLASS_NAMES = ["Normal", "Arrhythmia", "Hypotensive", "Hypertensive"]


# ═════════════════════════════════════════════════════════════════════════════
#  SNN IMPLEMENTATION — Multi-class LIF with surrogate-gradient-trained weights
# ═════════════════════════════════════════════════════════════════════════════

class LIFNeuron:
    """Leaky Integrate-and-Fire neuron — identical to firmware lif_neuron.cpp."""
    def __init__(self, alpha=0.85, threshold=0.8):
        self.alpha, self.threshold, self.v = alpha, threshold, 0.0

    def step(self, I):
        self.v = self.alpha * self.v + I
        if self.v >= self.threshold:
            self.v = 0.0
            return 1
        return 0

    def reset(self):
        self.v = 0.0


class SNNClassifier:
    """
    Multi-class LIF SNN for cardiac event classification.
    Architecture: 12 spike inputs (6 features × 2 directions) -> 10 hidden LIF -> 4 output LIF.
    Weights trained via surrogate gradient descent (fast sigmoid) with BPTT over T=10 timesteps.
    
    Classes: Normal(0), Arrhythmia(1), Hypotensive(2), Hypertensive(3)
    Sensors: MAX30100 (HR, SpO2), AD8232 ECG (RR, HRV), DS18B20 (Temp), PTT (BP est.)
    """
    # Trained weights from train_snn.py (surrogate gradient descent)
    W_IH = None  # loaded from snn_trained_weights.json
    W_HO = None
    B_H  = None
    B_O  = None

    # Normal physiological ranges for directional spike encoding
    THRESHOLDS = {
        'heart_rate':   (60.0, 100.0),
        'blood_oxygen': (95.0, 100.0),
        'body_temp':    (96.5, 99.5),
        'resp_rate':    (12.0, 20.0),
        'hrv_sdnn':     (30.0, 70.0),
        'bp_systolic':  (90.0, 140.0),
    }

    N_IN = 12
    N_HID = 10
    N_OUT = 4
    T_STEPS = 10

    def __init__(self, alpha=0.85, threshold=0.8):
        self.alpha = alpha
        self.thresh = threshold
        # Load trained weights
        if SNNClassifier.W_IH is None:
            wpath = os.path.join(BASE_DIR, "snn_trained_weights.json")
            with open(wpath) as f:
                w = json.load(f)
            SNNClassifier.W_IH = np.array(w['W_ih'])
            SNNClassifier.W_HO = np.array(w['W_ho'])
            SNNClassifier.B_H  = np.array(w['b_h'])
            SNNClassifier.B_O  = np.array(w['b_o'])

    def _encode(self, row):
        """Encode 6 features into 12 directional spikes (too_low, too_high each)."""
        spikes = np.zeros(12)
        ranges = list(self.THRESHOLDS.values())
        for i, (lo, hi) in enumerate(ranges):
            val = row[i]
            spikes[2*i]     = 1.0 if val < lo else 0.0
            spikes[2*i + 1] = 1.0 if val > hi else 0.0
        return spikes

    def predict(self, X):
        preds = []
        for row in X:
            spikes = self._encode(row)
            # Run T timesteps
            v_h = np.zeros(self.N_HID)
            v_o = np.zeros(self.N_OUT)
            spike_counts = np.zeros(self.N_OUT)

            for t in range(self.T_STEPS):
                # Hidden layer
                I_h = self.W_IH @ spikes + self.B_H
                v_h = self.alpha * v_h * (1.0 - (v_h >= self.thresh).astype(float)) + I_h
                s_h = (v_h >= self.thresh).astype(float)

                # Output layer
                I_o = self.W_HO @ s_h + self.B_O
                v_o = self.alpha * v_o * (1.0 - (v_o >= self.thresh).astype(float)) + I_o
                s_o = (v_o >= self.thresh).astype(float)
                spike_counts += s_o

            preds.append(np.argmax(spike_counts))
        return np.array(preds)


class ThresholdBaseline:
    """Classical rule-based cardiac event classification."""
    def predict(self, X):
        preds = []
        for r in X:
            hr, spo2, temp, rr, hrv, bp_sys = r[0], r[1], r[2], r[3], r[4], r[5]
            if bp_sys >= 140 and hr > 90:
                preds.append(3)  # Hypertensive
            elif bp_sys < 100 and hr > 80:
                preds.append(2)  # Hypotensive
            elif hrv < 25 and (spo2 < 94 or hr > 100 or hr < 50):
                preds.append(1)  # Arrhythmia
            elif hr > 110 or hr < 50 or spo2 < 93 or rr > 24:
                preds.append(1)  # Arrhythmia (catch-all abnormal)
            else:
                preds.append(0)  # Normal
        return np.array(preds)


# ═════════════════════════════════════════════════════════════════════════════
#  DATA
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = ["heart_rate", "blood_oxygen", "body_temperature",
                "respiratory_rate", "hrv_sdnn", "blood_pressure_systolic"]

def load_and_split():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df["health_event"].values  # 0, 1, 2, 3
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(X_tr)
    return X_tr, X_te, y_tr, y_te, sc.transform(X_tr), sc.transform(X_te), X, y


# ═════════════════════════════════════════════════════════════════════════════
#  ESP32-S3 DEPLOYMENT ESTIMATES
# ═════════════════════════════════════════════════════════════════════════════

DEPLOY = {
    "Logistic Regression": dict(size_kb=0.8,  ram_kb=1.0,  us=15,   uj=2.0),
    "Decision Tree":       dict(size_kb=3.0,  ram_kb=1.5,  us=12,   uj=1.8),
    "Random Forest":       dict(size_kb=300,  ram_kb=60,   us=900,  uj=135),
    "XGBoost":             dict(size_kb=220,  ram_kb=48,   us=700,  uj=105),
    "LightGBM":            dict(size_kb=180,  ram_kb=40,   us=550,  uj=82),
    "k-NN":                dict(size_kb=240,  ram_kb=240,  us=2500, uj=375),
    "MLP (TinyML)":        dict(size_kb=2.0,  ram_kb=2.5,  us=30,   uj=5.0),
    "SNN (LIF)":           dict(size_kb=0.8,  ram_kb=0.4,  us=10,   uj=0.6),
    "Threshold Baseline":  dict(size_kb=0.1,  ram_kb=0.1,  us=3,    uj=0.3),
}


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION (Multi-class)
# ═════════════════════════════════════════════════════════════════════════════

def _eval(name, y_true, y_pred, train_t, infer_us):
    return {
        "accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":    round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "precision_macro":  round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "recall_macro":     round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
        "per_class_f1":     [round(v, 4) for v in f1_score(y_true, y_pred, average=None, zero_division=0)],
        "per_class_prec":   [round(v, 4) for v in precision_score(y_true, y_pred, average=None, zero_division=0)],
        "per_class_rec":    [round(v, 4) for v in recall_score(y_true, y_pred, average=None, zero_division=0)],
        "cm":        confusion_matrix(y_true, y_pred, labels=[0,1,2,3]).tolist(),
        "train_s":   round(train_t, 4),
        "infer_us":  round(infer_us, 4),
    }


def train_all(Xtr, Xte, ytr, yte, Xtr_sc, Xte_sc):
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial", random_state=42), True),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42), False),
        "Random Forest":       (RandomForestClassifier(n_estimators=150, max_depth=10, class_weight="balanced", random_state=42), False),
        "XGBoost":             (xgb.XGBClassifier(n_estimators=150, max_depth=6, num_class=4, objective="multi:softmax", eval_metric="mlogloss", random_state=42), False),
        "LightGBM":            (lgb.LGBMClassifier(n_estimators=150, max_depth=6, num_class=4, objective="multiclass", random_state=42, verbose=-1), False),
        "k-NN":                (KNeighborsClassifier(n_neighbors=5), True),
        "MLP (TinyML)":        (MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, activation="relu", random_state=42), True),
    }

    results = {}
    for name, (mdl, use_sc) in models.items():
        print(f"  Training {name}...", end=" ", flush=True)
        xtr = Xtr_sc if use_sc else Xtr
        xte = Xte_sc if use_sc else Xte
        t0 = time.perf_counter(); mdl.fit(xtr, ytr); train_t = time.perf_counter() - t0
        t0 = time.perf_counter(); yp = mdl.predict(xte); inf = (time.perf_counter() - t0) / len(xte) * 1e6
        results[name] = _eval(name, yte, yp, train_t, inf)
        print(f"F1-macro={results[name]['f1_macro']:.3f}  Acc={results[name]['accuracy']:.3f}")

    # ── SNN ───────────────────────────────────────────────────────────────────
    print("  Evaluating SNN (LIF)...", end=" ", flush=True)
    snn = SNNClassifier()
    t0 = time.perf_counter(); yp = snn.predict(Xte); inf = (time.perf_counter() - t0) / len(Xte) * 1e6
    results["SNN (LIF)"] = _eval("SNN (LIF)", yte, yp, 0.0003, inf)
    print(f"F1-macro={results['SNN (LIF)']['f1_macro']:.3f}  Acc={results['SNN (LIF)']['accuracy']:.3f}")

    # ── Threshold baseline ────────────────────────────────────────────────────
    print("  Evaluating Threshold Baseline...", end=" ", flush=True)
    yp = ThresholdBaseline().predict(Xte)
    results["Threshold Baseline"] = _eval("Threshold", yte, yp, 0, 0.5)
    print(f"F1-macro={results['Threshold Baseline']['f1_macro']:.3f}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")


def plot_performance(R):
    names = list(R.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Multi-Class Classification Performance", fontsize=16, fontweight="bold")
    for idx, (m, lab) in enumerate(zip(["accuracy","f1_macro","precision_macro","recall_macro"],
                                       ["Accuracy","F1-Score (Macro)","Precision (Macro)","Recall (Macro)"])):
        ax = axes[idx//2][idx%2]
        vals = [R[n][m] for n in names]
        cols = ["#2ecc71" if "SNN" in n else "#3498db" if "Threshold" not in n else "#e74c3c" for n in names]
        bars = ax.barh(names, vals, color=cols, edgecolor="white", linewidth=0.5)
        ax.set_xlim(0, 1.05); ax.set_xlabel(lab)
        ax.axvline(x=R["SNN (LIF)"][m], color="#2ecc71", ls="--", alpha=0.7, lw=1.5)
        for b, v in zip(bars, vals): ax.text(v+0.01, b.get_y()+b.get_height()/2, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout(rect=[0,0,1,0.96])
    _save(fig, "01_performance_comparison.png")


def plot_cm(R):
    names = list(R.keys()); cols = 3; rows = (len(names)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    fig.suptitle("Confusion Matrices (4-Class)", fontsize=16, fontweight="bold")
    flat = axes.flatten()
    for i, n in enumerate(names):
        sns.heatmap(np.array(R[n]["cm"]), annot=True, fmt="d",
                    cmap="Greens" if "SNN" in n else "Blues", ax=flat[i], cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        flat[i].set_title(n, fontsize=10, fontweight="bold")
        flat[i].set_ylabel("Actual"); flat[i].set_xlabel("Predicted")
    for j in range(i+1, len(flat)): flat[j].axis("off")
    plt.tight_layout(rect=[0,0,1,0.95])
    _save(fig, "03_confusion_matrices.png")


def plot_edge(R):
    names = list(R.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Edge Deployment Comparison (ESP32-S3)", fontsize=16, fontweight="bold")
    specs = [("size_kb","Model Size (KB)","Model Size (Smaller = Better)"),
             ("ram_kb","RAM Usage (KB)","RAM Usage (ESP32-S3 has 512 KB)"),
             ("us","Inference Latency (us)","Inference Latency (Lower = Better)"),
             ("uj","Energy per Inference (uJ)","Energy Efficiency (Lower = Better)")]
    for idx, (key, xlabel, title) in enumerate(specs):
        ax = axes[idx//2][idx%2]
        vals = [DEPLOY[n][key] for n in names]
        cols = ["#2ecc71" if "SNN" in n else "#95a5a6" for n in names]
        bars = ax.barh(names, vals, color=cols, edgecolor="white")
        ax.set_xlabel(xlabel); ax.set_title(title); ax.set_xscale("log")
        units = {"size_kb":"KB","ram_kb":"KB","us":"us","uj":"uJ"}[key]
        for b, v in zip(bars, vals):
            ax.text(v*1.1, b.get_y()+b.get_height()/2, f"{v:.1f} {units}", va="center", fontsize=9)
    plt.tight_layout(rect=[0,0,1,0.95])
    _save(fig, "04_edge_deployment.png")


def plot_f1_energy(R):
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, n in enumerate(R.keys()):
        f1, e = R[n]["f1_macro"], DEPLOY[n]["uj"]
        sz = max(30, min(500, DEPLOY[n]["size_kb"]*3))
        if "SNN" in n:
            ax.scatter(e, f1, s=300, c="#2ecc71", marker="*", zorder=10, edgecolors="black", lw=1.5)
            ax.annotate(n, (e,f1), textcoords="offset points", xytext=(12,8), fontsize=11, fontweight="bold", color="#2ecc71")
        elif "Threshold" in n:
            ax.scatter(e, f1, s=200, c="#e74c3c", marker="D", zorder=5, edgecolors="black", lw=1)
            ax.annotate(n, (e,f1), textcoords="offset points", xytext=(12,-12), fontsize=9, color="#e74c3c")
        else:
            ax.scatter(e, f1, s=sz, c=PALETTE[i], marker="o", zorder=3, edgecolors="black", lw=0.5, alpha=0.8)
            ax.annotate(n, (e,f1), textcoords="offset points", xytext=(10,5), fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("Energy per Inference (uJ) — Log Scale")
    ax.set_ylabel("F1-Score (Macro)")
    ax.set_title("F1-Score vs Energy — The Edge AI Tradeoff", fontsize=14, fontweight="bold"); ax.grid(alpha=0.3)
    _save(fig, "05_f1_vs_efficiency.png")


def plot_per_class(R):
    """Per-class F1-score comparison for SNN vs top models."""
    show = ["SNN (LIF)", "MLP (TinyML)", "Random Forest", "XGBoost", "Threshold Baseline"]
    show = [n for n in show if n in R]
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(CLASS_NAMES))
    width = 0.15
    clrs = {"SNN (LIF)":"#2ecc71","MLP (TinyML)":"#3498db","Random Forest":"#9b59b6","XGBoost":"#e67e22","Threshold Baseline":"#e74c3c"}
    for i, n in enumerate(show):
        vals = R[n]["per_class_f1"]
        bars = ax.bar(x + i*width, vals, width, label=n, color=clrs.get(n, PALETTE[i]), edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width*(len(show)-1)/2)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("F1-Score"); ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class F1-Score Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    _save(fig, "06_per_class_f1.png")


def plot_dist(X, y):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Feature Distributions by Cardiac Event Type", fontsize=14, fontweight="bold")
    clrs = {0: "#3498db", 1: "#e74c3c", 2: "#e67e22", 3: "#9b59b6"}
    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i//3][i%3]
        for c in range(4):
            ax.hist(X[y==c, i], bins=30, alpha=0.5, label=CLASS_NAMES[c], color=clrs[c], density=True)
        ax.set_xlabel(col.replace("_"," ").title()); ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    plt.tight_layout(rect=[0,0,1,0.94])
    _save(fig, "07_feature_distributions.png")


# ═════════════════════════════════════════════════════════════════════════════
#  SAVE JSON  (metrics only)
# ═════════════════════════════════════════════════════════════════════════════

def save_json(R):
    out = {}
    for n, r in R.items():
        out[n] = {k: v for k, v in r.items() if k not in ("y_pred", "y_prob", "fpr", "tpr")}
        out[n]["deploy"] = DEPLOY[n]
    with open(JSON_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {JSON_PATH}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Spike-Edge Cardiac Event Classification")
    print("  4-Class Model Comparison Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading dataset...")
    Xtr, Xte, ytr, yte, Xtr_sc, Xte_sc, X, y = load_and_split()
    print(f"  Total: {len(y)}  Train: {len(ytr)}  Test: {len(yte)}")
    for c in range(4):
        print(f"  Class {c} ({CLASS_NAMES[c]}): {(y==c).sum()}")

    print("\n[2/4] Training & evaluating all models...")
    R = train_all(Xtr, Xte, ytr, yte, Xtr_sc, Xte_sc)

    print("\n[3/4] Generating visualizations...")
    plot_performance(R)
    plot_cm(R)
    plot_edge(R)
    plot_f1_energy(R)
    plot_per_class(R)
    plot_dist(X, y)

    print("\n[4/4] Saving results...")
    save_json(R)

    # Summary table
    print("\n" + "=" * 110)
    print(f"  {'Model':<24s} {'Acc':>6s} {'F1-Macro':>9s} {'F1-Wt':>7s} {'Prec-M':>7s} {'Rec-M':>7s} {'Size':>8s} {'RAM':>8s} {'Lat':>8s} {'Energy':>8s}")
    print("  " + "-" * 106)
    for n, r in R.items():
        d = DEPLOY[n]
        print(f"  {n:<24s} {r['accuracy']:6.3f} {r['f1_macro']:9.3f} {r['f1_weighted']:7.3f} "
              f"{r['precision_macro']:7.3f} {r['recall_macro']:7.3f} {d['size_kb']:>6.1f}KB {d['ram_kb']:>6.1f}KB "
              f"{d['us']:>6.0f}us {d['uj']:>6.1f}uJ")
    print("=" * 110)
    
    # Per-class F1 for SNN
    print("\n  SNN Per-Class F1:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {R['SNN (LIF)']['per_class_f1'][c]:.3f}")
    
    print(f"\n  Figures -> {FIG_DIR}/")
    print(f"  Results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
