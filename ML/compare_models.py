"""
compare_models.py
Train and compare TinyML-class models against our SNN approach for
cardiac anomaly detection using the IoT health monitoring dataset.

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
    confusion_matrix, roc_curve, auc, average_precision_score,
    precision_recall_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
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


# ═════════════════════════════════════════════════════════════════════════════
#  SNN IMPLEMENTATION  (matches firmware LIF architecture)
# ═════════════════════════════════════════════════════════════════════════════

class LIFNeuron:
    """Leaky Integrate-and-Fire neuron — identical to firmware lif_neuron.cpp."""
    def __init__(self, alpha=0.9, threshold=1.0):
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
    LIF-based SNN with deviation-from-normal spike encoding.
    Architecture: 4 inputs -> 5 hidden LIF -> 1 output LIF.
    Weights match firmware config.h exactly.
    """
    W_IH = np.array([
        [0.50, 0.40, 0.20, 0.30],
        [0.45, 0.35, 0.25, 0.20],
        [0.40, 0.50, 0.15, 0.35],
        [0.35, 0.30, 0.30, 0.40],
        [0.55, 0.45, 0.10, 0.25],
    ])
    W_HO = np.array([0.40, 0.35, 0.30, 0.25, 0.45])

    # Normal physiological ranges for deviation encoding
    NORMAL_HR   = (60.0, 100.0)
    NORMAL_SPO2 = (95.0, 100.0)
    NORMAL_TEMP = (96.5, 99.5)   # dataset uses Fahrenheit
    NORMAL_RR   = (12.0, 20.0)

    def __init__(self, alpha=0.9, threshold=1.0):
        self.hidden = [LIFNeuron(alpha, threshold) for _ in range(5)]
        self.output = LIFNeuron(alpha, threshold)

    def _encode(self, hr, spo2, temp, rr):
        return [
            1 if (hr < self.NORMAL_HR[0] or hr > self.NORMAL_HR[1]) else 0,
            1 if spo2 < self.NORMAL_SPO2[0] else 0,
            1 if (temp < self.NORMAL_TEMP[0] or temp > self.NORMAL_TEMP[1]) else 0,
            1 if (rr < self.NORMAL_RR[0] or rr > self.NORMAL_RR[1]) else 0,
        ]

    def _forward(self, spikes):
        h = [self.hidden[i].step(sum(self.W_IH[i][j] * spikes[j] for j in range(4)))
             for i in range(5)]
        return self.output.step(sum(self.W_HO[i] * h[i] for i in range(5)))

    def reset(self):
        for n in self.hidden: n.reset()
        self.output.reset()

    def predict(self, X):
        self.reset()
        preds = []
        for row in X:
            spikes = self._encode(row[0], row[1], row[2], row[3] if len(row) > 3 else 16.0)
            preds.append(self._forward(spikes))
        return np.array(preds)


class ThresholdBaseline:
    """Classical rule-based anomaly detection."""
    def predict(self, X):
        return np.array([
            int(r[0] > 100 or r[0] < 50 or r[1] < 94 or r[2] > 99.5 or r[2] < 96.0)
            for r in X
        ])


# ═════════════════════════════════════════════════════════════════════════════
#  DATA
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = ["heart_rate", "blood_oxygen", "body_temperature",
                "respiratory_rate", "hrv_sdnn"]

def load_and_split():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = (df["health_event"] > 0).astype(int).values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(X_tr)
    return X_tr, X_te, y_tr, y_te, sc.transform(X_tr), sc.transform(X_te), X, y


# ═════════════════════════════════════════════════════════════════════════════
#  ESP32-S3 DEPLOYMENT ESTIMATES
# ═════════════════════════════════════════════════════════════════════════════

DEPLOY = {
    "Logistic Regression": dict(size_kb=0.2,  ram_kb=0.5,  us=5,    uj=0.8),
    "Decision Tree":       dict(size_kb=2.5,  ram_kb=1.0,  us=10,   uj=1.5),
    "Random Forest":       dict(size_kb=250,  ram_kb=50,   us=800,  uj=120),
    "XGBoost":             dict(size_kb=180,  ram_kb=40,   us=600,  uj=90),
    "LightGBM":            dict(size_kb=150,  ram_kb=35,   us=500,  uj=75),
    "k-NN":                dict(size_kb=200,  ram_kb=200,  us=2000, uj=300),
    "MLP (TinyML)":        dict(size_kb=1.5,  ram_kb=2.0,  us=25,   uj=4.0),
    "SNN (LIF)":           dict(size_kb=0.3,  ram_kb=0.2,  us=8,    uj=0.5),
    "Threshold Baseline":  dict(size_kb=0.05, ram_kb=0.1,  us=2,    uj=0.2),
}


# ═════════════════════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def _eval(name, y_true, y_pred, y_prob, train_t, infer_us):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(auc(fpr, tpr), 4),
        "cm":        confusion_matrix(y_true, y_pred).tolist(),
        "fpr": fpr, "tpr": tpr,
        "y_pred": y_pred, "y_prob": y_prob,
        "train_s": round(train_t, 4),
        "infer_us": round(infer_us, 4),
    }


def train_all(Xtr, Xte, ytr, yte, Xtr_sc, Xte_sc):
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42), True),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42), False),
        "Random Forest":       (RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42), False),
        "XGBoost":             (xgb.XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=5.67, eval_metric="logloss", random_state=42), False),
        "LightGBM":            (lgb.LGBMClassifier(n_estimators=100, max_depth=5, scale_pos_weight=5.67, random_state=42, verbose=-1), False),
        "k-NN":                (KNeighborsClassifier(n_neighbors=5), True),
        "MLP (TinyML)":        (MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, activation="relu", random_state=42), True),
    }

    results = {}
    for name, (mdl, use_sc) in models.items():
        print(f"  Training {name}...", end=" ", flush=True)
        xtr = Xtr_sc if use_sc else Xtr
        xte = Xte_sc if use_sc else Xte
        t0 = time.perf_counter(); mdl.fit(xtr, ytr); train_t = time.perf_counter() - t0
        t0 = time.perf_counter(); yp = mdl.predict(xte); inf = (time.perf_counter() - t0) / len(xte) * 1e6
        prob = mdl.predict_proba(xte)[:, 1] if hasattr(mdl, "predict_proba") else yp.astype(float)
        results[name] = _eval(name, yte, yp, prob, train_t, inf)
        print(f"F1={results[name]['f1']:.3f}  AUC={results[name]['roc_auc']:.3f}")

    # ── SNN with LIF parameter tuning ────────────────────────────────────────
    print("  Evaluating SNN (LIF)...", end=" ", flush=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    ti, vi = next(sss.split(Xtr, ytr))
    best, ba, bt = 0, 0.9, 1.0
    for a in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]:
        for t in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2]:
            f = f1_score(ytr[vi], SNNClassifier(a, t).predict(Xtr[vi]), zero_division=0)
            if f > best: best, ba, bt = f, a, t
    snn = SNNClassifier(ba, bt)
    t0 = time.perf_counter(); yp = snn.predict(Xte); inf = (time.perf_counter() - t0) / len(Xte) * 1e6
    results["SNN (LIF)"] = _eval("SNN (LIF)", yte, yp, yp.astype(float), 0, inf)
    print(f"F1={results['SNN (LIF)']['f1']:.3f} (alpha={ba}, thr={bt})")

    # ── Threshold baseline ────────────────────────────────────────────────────
    print("  Evaluating Threshold Baseline...", end=" ", flush=True)
    yp = ThresholdBaseline().predict(Xte)
    results["Threshold Baseline"] = _eval("Threshold", yte, yp, yp.astype(float), 0, 0.5)
    print(f"F1={results['Threshold Baseline']['f1']:.3f}")

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
    fig.suptitle("Classification Performance Comparison", fontsize=16, fontweight="bold")
    for idx, (m, lab) in enumerate(zip(["accuracy","precision","recall","f1"],
                                       ["Accuracy","Precision","Recall","F1-Score"])):
        ax = axes[idx//2][idx%2]
        vals = [R[n][m] for n in names]
        cols = ["#2ecc71" if "SNN" in n else "#3498db" if "Threshold" not in n else "#e74c3c" for n in names]
        bars = ax.barh(names, vals, color=cols, edgecolor="white", linewidth=0.5)
        ax.set_xlim(0, 1.05); ax.set_xlabel(lab)
        ax.axvline(x=R["SNN (LIF)"][m], color="#2ecc71", ls="--", alpha=0.7, lw=1.5)
        for b, v in zip(bars, vals): ax.text(v+0.01, b.get_y()+b.get_height()/2, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout(rect=[0,0,1,0.96])
    _save(fig, "01_performance_comparison.png")


def plot_roc(R):
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, r in R.items():
        fpr, tpr = r["fpr"], r["tpr"]
        if len(fpr) > 2:
            ax.plot(fpr, tpr, label=f'{name} (AUC={r["roc_auc"]:.3f})',
                    lw=2.5 if "SNN" in name else 1.5, alpha=1.0 if "SNN" in name else 0.7)
        else:
            ax.scatter([fpr[-1]], [tpr[-1]], label=f'{name} (AUC={r["roc_auc"]:.3f})',
                       s=120, zorder=5, marker="*" if "SNN" in name else "o")
    ax.plot([0,1],[0,1],"k--",alpha=0.3,label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    _save(fig, "02_roc_curves.png")


def plot_cm(R):
    names = list(R.keys()); cols = 3; rows = (len(names)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")
    flat = axes.flatten()
    for i, n in enumerate(names):
        sns.heatmap(np.array(R[n]["cm"]), annot=True, fmt="d",
                    cmap="Greens" if "SNN" in n else "Blues", ax=flat[i], cbar=False,
                    xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"])
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
        if key == "ram_kb":
            ax.axvline(x=512, color="red", ls="--", alpha=0.5, label="ESP32-S3 limit (512 KB)")
            ax.legend()
    plt.tight_layout(rect=[0,0,1,0.95])
    _save(fig, "04_edge_deployment.png")


def plot_f1_energy(R):
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, n in enumerate(R.keys()):
        f1, e = R[n]["f1"], DEPLOY[n]["uj"]
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
    ax.set_xscale("log"); ax.set_xlabel("Energy per Inference (uJ) — Log Scale", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("F1-Score vs Energy Efficiency — The Edge AI Tradeoff", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    _save(fig, "05_f1_vs_efficiency.png")


def plot_radar(R):
    cats = ["F1-Score","Precision","Recall","Memory\nEfficiency","Energy\nEfficiency","Latency\nScore"]
    def scores(n):
        r, d = R[n], DEPLOY[n]
        me = 1-(np.log10(max(d["ram_kb"],0.1))/np.log10(300))
        ee = 1-(np.log10(max(d["uj"],0.1))/np.log10(400))
        le = 1-(np.log10(max(d["us"],1))/np.log10(3000))
        return [r["f1"],r["precision"],r["recall"],max(0,min(1,me)),max(0,min(1,ee)),max(0,min(1,le))]
    show = ["SNN (LIF)","MLP (TinyML)","Random Forest","XGBoost","Threshold Baseline"]
    angles = np.linspace(0,2*np.pi,len(cats),endpoint=False).tolist(); angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(polar=True))
    clrs = {"SNN (LIF)":"#2ecc71","MLP (TinyML)":"#3498db","Random Forest":"#9b59b6","XGBoost":"#e67e22","Threshold Baseline":"#e74c3c"}
    for n in show:
        if n not in R: continue
        s = scores(n); s += s[:1]
        ax.plot(angles, s, lw=3 if "SNN" in n else 1.5, label=n, color=clrs.get(n))
        ax.fill(angles, s, alpha=0.3 if "SNN" in n else 0.1, color=clrs.get(n))
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=11); ax.set_ylim(0,1)
    ax.set_title("Multi-Dimensional Model Comparison\n(SNN vs Industry Models)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1), fontsize=10); ax.grid(alpha=0.3)
    _save(fig, "06_radar_comparison.png")


def plot_dist(X, y):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("Feature Distributions: Normal vs Anomaly", fontsize=14, fontweight="bold")
    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        ax.hist(X[y==0,i], bins=30, alpha=0.6, label="Normal", color="#3498db", density=True)
        ax.hist(X[y==1,i], bins=30, alpha=0.6, label="Anomaly", color="#e74c3c", density=True)
        ax.set_xlabel(col.replace("_"," ").title()); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    plt.tight_layout(rect=[0,0,1,0.94])
    _save(fig, "07_feature_distributions.png")


def plot_size_acc(R):
    fig, ax = plt.subplots(figsize=(12, 8))
    for n in R:
        acc, sz, ram = R[n]["accuracy"], DEPLOY[n]["size_kb"], DEPLOY[n]["ram_kb"]
        if "SNN" in n:
            ax.scatter(sz, acc, s=400, c="#2ecc71", marker="*", zorder=10, edgecolors="black", lw=2)
            ax.annotate(n, (sz,acc), textcoords="offset points", xytext=(15,8), fontsize=12, fontweight="bold", color="#2ecc71")
        elif "Threshold" in n:
            ax.scatter(sz, acc, s=200, c="#e74c3c", marker="D", zorder=5, edgecolors="black", lw=1)
            ax.annotate(n, (sz,acc), textcoords="offset points", xytext=(10,-15), fontsize=9, color="#e74c3c")
        else:
            ax.scatter(sz, acc, s=max(50,ram*5), c="#3498db", marker="o", alpha=0.7, edgecolors="black", lw=0.5)
            ax.annotate(n, (sz,acc), textcoords="offset points", xytext=(10,5), fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("Model Size (KB) — Log Scale", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Size vs Accuracy (bubble ~ RAM usage)", fontsize=14, fontweight="bold"); ax.grid(alpha=0.3)
    _save(fig, "08_size_vs_accuracy.png")


# ═════════════════════════════════════════════════════════════════════════════
#  SAVE JSON  (metrics only — no numpy arrays)
# ═════════════════════════════════════════════════════════════════════════════

def save_json(R):
    out = {}
    for n, r in R.items():
        out[n] = {k: v for k, v in r.items() if k in ("accuracy","precision","recall","f1","roc_auc","cm","train_s","infer_us")}
        out[n]["deploy"] = DEPLOY[n]
    with open(JSON_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {JSON_PATH}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Spike-Edge Cardiac Anomaly Detection")
    print("  Model Comparison Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading dataset...")
    Xtr, Xte, ytr, yte, Xtr_sc, Xte_sc, X, y = load_and_split()
    print(f"  Total: {len(y)}  Train: {len(ytr)}  Test: {len(yte)}")
    print(f"  Normal: {(y==0).sum()}  Anomaly: {(y==1).sum()}")

    print("\n[2/4] Training & evaluating all models...")
    R = train_all(Xtr, Xte, ytr, yte, Xtr_sc, Xte_sc)

    print("\n[3/4] Generating visualizations...")
    plot_performance(R)
    plot_roc(R)
    plot_cm(R)
    plot_edge(R)
    plot_f1_energy(R)
    plot_radar(R)
    plot_dist(X, y)
    plot_size_acc(R)

    print("\n[4/4] Saving results...")
    save_json(R)

    # Summary table
    print("\n" + "=" * 100)
    print(f"  {'Model':<24s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s} {'Size':>8s} {'RAM':>8s} {'Lat':>8s} {'Energy':>8s}")
    print("  " + "-" * 96)
    for n, r in R.items():
        d = DEPLOY[n]
        print(f"  {n:<24s} {r['accuracy']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} "
              f"{r['f1']:6.3f} {r['roc_auc']:6.3f} {d['size_kb']:>6.1f}KB {d['ram_kb']:>6.1f}KB "
              f"{d['us']:>6.0f}us {d['uj']:>6.1f}uJ")
    print("=" * 100)
    print(f"  Figures -> {FIG_DIR}/")
    print(f"  Results -> {JSON_PATH}")


if __name__ == "__main__":
    main()
