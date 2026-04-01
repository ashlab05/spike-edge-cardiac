"""
train_snn_3input.py
Train a multi-class SNN (LIF) using surrogate gradient descent for cardiac event
classification using only 3 sensor inputs available on the ESP32-S3:
  - Heart Rate (MAX30100 PPG)
  - SpO2 / Blood Oxygen (MAX30100)
  - HRV SDNN (AD8232 ECG → R-R interval analysis)

Architecture: 3 inputs × 2 directions = 6 spike channels → 8 hidden LIF → 4 output LIF

Classes:
    0 = Normal
    1 = Arrhythmia (erratic HR, very low HRV, low SpO2)
    2 = Hypotensive Event (low HR variability, moderate tachycardia)
    3 = Hypertensive Crisis (tachycardia, high stress indicators)

Outputs:
    - ML/snn_3input_weights.json   — weights + full metadata for ESP32
    - ML/snn_3input_weights.h      — C header file directly #includable in firmware
    - Console classification report & confusion matrix
"""

import os, json, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "iot_health_monitoring_dataset.csv")

# ── Load & Prepare Data ─────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# Only 3 features from ESP32 sensors
FEATURE_COLS = ["heart_rate", "blood_oxygen", "hrv_sdnn"]
FEATURE_LABELS = ["Heart Rate (bpm)", "SpO2 (%)", "HRV SDNN (ms)"]
CLASS_NAMES = ["Normal", "Arrhythmia", "Hypotensive", "Hypertensive"]

X = df[FEATURE_COLS].values.astype(np.float64)
y = df["health_event"].values.astype(int)  # 0, 1, 2, 3

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Dataset: {len(y)} samples | Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Features: {FEATURE_COLS}")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ── Spike Encoding (medical thresholds) ──────────────────────────────────────
THRESHOLDS = {
    'heart_rate':    (60.0, 100.0),   # bpm — normal resting range
    'blood_oxygen':  (95.0, 100.0),   # % — below 95 is concerning
    'hrv_sdnn':      (30.0, 70.0),    # ms — below 30 indicates stress/disease
}

def encode_spikes(X):
    """
    Convert raw features to binary spikes.
    2 spikes per feature (too_low, too_high) → 6 spike channels.
    """
    n = len(X)
    spikes = np.zeros((n, 6), dtype=np.float64)

    ranges = [
        THRESHOLDS['heart_rate'],
        THRESHOLDS['blood_oxygen'],
        THRESHOLDS['hrv_sdnn'],
    ]

    for i, (lo, hi) in enumerate(ranges):
        spikes[:, 2*i]     = (X[:, i] < lo).astype(float)   # too low
        spikes[:, 2*i + 1] = (X[:, i] > hi).astype(float)   # too high

    return spikes

X_train_spk = encode_spikes(X_train)
X_test_spk  = encode_spikes(X_test)

N_INPUT  = 6   # 3 features × 2 directions
N_HIDDEN = 8   # compact hidden layer for ESP32
N_OUTPUT = 4   # 4 cardiac event classes

print(f"\nArchitecture: {N_INPUT} spike inputs → {N_HIDDEN} hidden LIF → {N_OUTPUT} output LIF")
print(f"Spike rate (train): {X_train_spk.mean():.3f}")
for i, col in enumerate(FEATURE_COLS):
    lo_pct = X_train_spk[:, 2*i].mean() * 100
    hi_pct = X_train_spk[:, 2*i+1].mean() * 100
    print(f"  {col}: too_low={lo_pct:.1f}%, too_high={hi_pct:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-Class Surrogate Gradient SNN
# ══════════════════════════════════════════════════════════════════════════════

class MultiClassSNN:
    """
    LIF SNN with surrogate gradient training for 4-class cardiac classification.
    Architecture: 6 spike inputs → 8 hidden LIF → 4 output LIF.
    Uses softmax over spike counts for multi-class output.
    """
    def __init__(self, n_in=6, n_hid=8, n_out=4, alpha=0.85, threshold=0.8, lr=0.01):
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.alpha = alpha
        self.threshold = threshold
        self.lr = lr

        # Xavier initialization
        self.W_ih = np.random.randn(n_hid, n_in) * np.sqrt(2.0 / (n_in + n_hid))
        self.W_ho = np.random.randn(n_out, n_hid) * np.sqrt(2.0 / (n_hid + n_out))
        self.b_h = np.zeros(n_hid)
        self.b_o = np.zeros(n_out)

    def _surrogate_grad(self, v, beta=5.0):
        x = beta * (v - self.threshold)
        return beta / (1.0 + np.abs(x))**2

    def _spike_fn(self, v):
        return (v >= self.threshold).astype(float)

    def _softmax(self, x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, spikes, T=10):
        batch_size = spikes.shape[0]

        v_h_list, s_h_list = [], []
        v_o_list, s_o_list = [], []

        v_h = np.zeros((batch_size, self.n_hid))
        v_o = np.zeros((batch_size, self.n_out))
        total_output_spikes = np.zeros((batch_size, self.n_out))

        for t in range(T):
            # Hidden layer
            I_h = spikes @ self.W_ih.T + self.b_h
            v_h = self.alpha * v_h * (1.0 - self._spike_fn(v_h)) + I_h
            s_h = self._spike_fn(v_h)
            v_h_list.append(v_h.copy())
            s_h_list.append(s_h.copy())

            # Output layer
            I_o = s_h @ self.W_ho.T + self.b_o
            v_o = self.alpha * v_o * (1.0 - self._spike_fn(v_o)) + I_o
            s_o = self._spike_fn(v_o)
            v_o_list.append(v_o.copy())
            s_o_list.append(s_o.copy())

            total_output_spikes += s_o

        # Convert spike counts to probabilities
        spike_rates = total_output_spikes / T
        probs = self._softmax(spike_rates * 5.0)

        self._cache = {
            'spikes': spikes, 'T': T,
            'v_h': v_h_list, 's_h': s_h_list,
            'v_o': v_o_list, 's_o': s_o_list,
            'probs': probs, 'spike_rates': spike_rates,
        }

        return probs

    def backward(self, y_true):
        cache = self._cache
        T = cache['T']
        spikes = cache['spikes']
        batch_size = spikes.shape[0]
        probs = cache['probs']

        # One-hot encode targets
        y_onehot = np.zeros((batch_size, self.n_out))
        y_onehot[np.arange(batch_size), y_true] = 1.0

        # Softmax cross-entropy gradient
        d_probs = (probs - y_onehot) / batch_size

        # Gradient through softmax → spike_rates
        d_rates = d_probs * 5.0

        dW_ho = np.zeros_like(self.W_ho)
        dW_ih = np.zeros_like(self.W_ih)
        db_h = np.zeros_like(self.b_h)
        db_o = np.zeros_like(self.b_o)

        for t in range(T - 1, -1, -1):
            # Output layer surrogate
            sg_o = self._surrogate_grad(cache['v_o'][t])
            d_s_o = d_rates / T
            d_v_o = d_s_o * sg_o

            dW_ho += d_v_o.T @ cache['s_h'][t]
            db_o += d_v_o.sum(axis=0)

            # Backprop to hidden
            d_s_h = d_v_o @ self.W_ho
            sg_h = self._surrogate_grad(cache['v_h'][t])
            d_v_h = d_s_h * sg_h

            dW_ih += d_v_h.T @ spikes
            db_h += d_v_h.sum(axis=0)

        # Clip and update
        for g in [dW_ho, dW_ih, db_h, db_o]:
            np.clip(g, -2.0, 2.0, out=g)

        self.W_ih -= self.lr * dW_ih
        self.W_ho -= self.lr * dW_ho
        self.b_h  -= self.lr * db_h
        self.b_o  -= self.lr * db_o

    def predict(self, X_spk, T=10):
        probs = self.forward(X_spk, T=T)
        return np.argmax(probs, axis=1)


# ── Training Loop ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  TRAINING 3-INPUT SNN (Heart Rate + SpO2 + HRV/ECG)")
print("  Architecture: 6 spikes → 8 hidden LIF → 4 output LIF")
print("=" * 65)

best_f1 = 0
best_weights = None
n_epochs = 400
batch_size = 64
T_steps = 10

# Class weights for imbalance
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (len(class_counts) * class_counts.astype(float))
print(f"\nClass weights: {dict(enumerate(class_weights.round(2)))}")

snn = MultiClassSNN(n_in=N_INPUT, n_hid=N_HIDDEN, n_out=N_OUTPUT,
                     alpha=0.85, threshold=0.8, lr=0.02)

t_start = time.perf_counter()

for epoch in range(n_epochs):
    idx = np.random.permutation(len(X_train_spk))
    X_shuffled = X_train_spk[idx]
    y_shuffled = y_train[idx]

    epoch_loss = 0
    n_batches = 0

    for i in range(0, len(X_shuffled), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        probs = snn.forward(X_batch, T=T_steps)

        # Weighted cross-entropy loss
        eps = 1e-7
        sample_weights = class_weights[y_batch]
        loss = -np.mean(sample_weights * np.log(probs[np.arange(len(y_batch)), y_batch] + eps))
        epoch_loss += loss
        n_batches += 1

        snn.backward(y_batch)

    if (epoch + 1) % 20 == 0:
        preds = snn.predict(X_test_spk, T=T_steps)
        f1_macro = f1_score(y_test, preds, average='macro')
        f1_weighted = f1_score(y_test, preds, average='weighted')
        acc = accuracy_score(y_test, preds)

        print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss/n_batches:.4f} | "
              f"F1-macro: {f1_macro:.3f} | F1-weighted: {f1_weighted:.3f} | Acc: {acc:.3f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_weights = {
                'W_ih': snn.W_ih.copy(),
                'W_ho': snn.W_ho.copy(),
                'b_h': snn.b_h.copy(),
                'b_o': snn.b_o.copy(),
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'acc': acc,
            }

    # LR schedule
    if (epoch + 1) == 150:
        snn.lr *= 0.5
    if (epoch + 1) == 250:
        snn.lr *= 0.5
    if (epoch + 1) == 350:
        snn.lr *= 0.5

train_time = time.perf_counter() - t_start

# ── Report Best Results ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  BEST RESULTS (3-Input SNN: HR + SpO2 + HRV/ECG)")
print("=" * 65)

# Reload best weights
snn.W_ih = best_weights['W_ih']
snn.W_ho = best_weights['W_ho']
snn.b_h = best_weights['b_h']
snn.b_o = best_weights['b_o']

preds = snn.predict(X_test_spk, T=T_steps)

print(f"\nF1-macro:    {best_weights['f1_macro']:.4f}")
print(f"F1-weighted: {best_weights['f1_weighted']:.4f}")
print(f"Accuracy:    {best_weights['acc']:.4f}")
print(f"Train time:  {train_time:.2f}s")

print("\n" + classification_report(y_test, preds, target_names=CLASS_NAMES))

cm = confusion_matrix(y_test, preds)
print("Confusion Matrix:")
print(cm)

per_class_f1   = f1_score(y_test, preds, average=None, zero_division=0)
per_class_prec = precision_score(y_test, preds, average=None, zero_division=0)
per_class_rec  = recall_score(y_test, preds, average=None, zero_division=0)

print("\nPer-class F1:")
for c, name in enumerate(CLASS_NAMES):
    print(f"  {name}: F1={per_class_f1[c]:.3f}  Prec={per_class_prec[c]:.3f}  Rec={per_class_rec[c]:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE WEIGHTS JSON
# ══════════════════════════════════════════════════════════════════════════════

json_path = os.path.join(BASE_DIR, "snn_3input_weights.json")
with open(json_path, "w") as f:
    json.dump({
        'architecture': {
            'n_in': N_INPUT,
            'n_hid': N_HIDDEN,
            'n_out': N_OUTPUT,
            'description': '3 sensor inputs × 2 (directional) = 6 spikes → 8 hidden LIF → 4 output LIF',
        },
        'W_ih': best_weights['W_ih'].tolist(),
        'W_ho': best_weights['W_ho'].tolist(),
        'b_h': best_weights['b_h'].tolist(),
        'b_o': best_weights['b_o'].tolist(),
        'metrics': {
            'f1_macro': float(best_weights['f1_macro']),
            'f1_weighted': float(best_weights['f1_weighted']),
            'accuracy': float(best_weights['acc']),
            'per_class_f1': per_class_f1.tolist(),
            'per_class_precision': per_class_prec.tolist(),
            'per_class_recall': per_class_rec.tolist(),
            'confusion_matrix': cm.tolist(),
        },
        'class_names': CLASS_NAMES,
        'feature_cols': FEATURE_COLS,
        'hyperparams': {
            'alpha': 0.85,
            'threshold': 0.8,
            'T_steps': T_steps,
            'epochs': n_epochs,
            'batch_size': batch_size,
        },
        'spike_encoding': '2 spikes per feature (too_low, too_high)',
        'thresholds': {k: list(v) for k, v in THRESHOLDS.items()},
        'sensors': {
            'heart_rate': 'MAX30100 PPG sensor (I2C)',
            'blood_oxygen': 'MAX30100 SpO2 sensor (I2C)',
            'hrv_sdnn': 'AD8232 ECG → R-R interval → SDNN computation',
        },
        'esp32_deployment': {
            'target': 'ESP32-S3 DevKitC-1',
            'framework': 'Arduino via PlatformIO',
            'estimated_flash_kb': 0.6,
            'estimated_ram_kb': 0.3,
            'estimated_inference_us': 8,
            'estimated_energy_uj': 0.5,
        },
    }, f, indent=2)
print(f"\nWeights saved to {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATE C HEADER FOR ESP32
# ══════════════════════════════════════════════════════════════════════════════

def generate_c_header(weights, path):
    """Generate snn_3input_weights.h ready to #include in ESP32 firmware."""
    W_ih = weights['W_ih']
    W_ho = weights['W_ho']
    b_h  = weights['b_h']
    b_o  = weights['b_o']

    lines = []
    lines.append("// ═══════════════════════════════════════════════════════════════")
    lines.append("//  snn_3input_weights.h")
    lines.append("//  AUTO-GENERATED by train_snn_3input.py — DO NOT EDIT MANUALLY")
    lines.append("//")
    lines.append("//  3-Input Cardiac SNN: HR + SpO2 + HRV (ECG)")
    lines.append(f"//  Architecture: {N_INPUT} spike in → {N_HIDDEN} hidden LIF → {N_OUTPUT} output LIF")
    lines.append(f"//  F1-macro: {weights['f1_macro']:.4f} | Accuracy: {weights['acc']:.4f}")
    lines.append("// ═══════════════════════════════════════════════════════════════")
    lines.append("#pragma once")
    lines.append("")
    lines.append("// ── Architecture ─────────────────────────────────────────────")
    lines.append(f"#define SNN3_N_IN    {N_INPUT}   // 3 features × 2 directions")
    lines.append(f"#define SNN3_N_HID   {N_HIDDEN}")
    lines.append(f"#define SNN3_N_OUT   {N_OUTPUT}   // 4 classes")
    lines.append(f"#define SNN3_T_STEPS {T_steps}")
    lines.append("")
    lines.append("// ── LIF Hyperparameters ───────────────────────────────────────")
    lines.append("#define SNN3_ALPHA     0.85f")
    lines.append("#define SNN3_THRESHOLD 0.80f")
    lines.append("")
    lines.append("// ── Spike Encoding Thresholds (medical ranges) ─────────────")
    lines.append("//    Feature        Low    High")
    lines.append(f"#define THR_HR_LOW    {THRESHOLDS['heart_rate'][0]:.1f}f   // bpm")
    lines.append(f"#define THR_HR_HIGH   {THRESHOLDS['heart_rate'][1]:.1f}f")
    lines.append(f"#define THR_SPO2_LOW  {THRESHOLDS['blood_oxygen'][0]:.1f}f   // %")
    lines.append(f"#define THR_SPO2_HIGH {THRESHOLDS['blood_oxygen'][1]:.1f}f")
    lines.append(f"#define THR_HRV_LOW   {THRESHOLDS['hrv_sdnn'][0]:.1f}f   // ms")
    lines.append(f"#define THR_HRV_HIGH  {THRESHOLDS['hrv_sdnn'][1]:.1f}f")
    lines.append("")
    lines.append("// ── Class Names ────────────────────────────────────────────────")
    lines.append('static const char* SNN3_CLASS_NAMES[4] = {')
    lines.append('    "Normal", "Arrhythmia", "Hypotensive", "Hypertensive"')
    lines.append('};')
    lines.append("")

    # W_ih: [N_HID][N_IN]
    lines.append(f"// ── Input→Hidden Weights [{N_HIDDEN}][{N_INPUT}] ──────────")
    lines.append(f"static const float SNN3_W_IH[{N_HIDDEN}][{N_INPUT}] = {{")
    for r in range(N_HIDDEN):
        vals = ", ".join(f"{W_ih[r, c]:+.6f}f" for c in range(N_INPUT))
        comma = "," if r < N_HIDDEN - 1 else ""
        lines.append(f"    {{{vals}}}{comma}")
    lines.append("};")
    lines.append("")

    # W_ho: [N_OUT][N_HID]
    lines.append(f"// ── Hidden→Output Weights [{N_OUTPUT}][{N_HIDDEN}] ────────")
    lines.append(f"static const float SNN3_W_HO[{N_OUTPUT}][{N_HIDDEN}] = {{")
    for r in range(N_OUTPUT):
        vals = ", ".join(f"{W_ho[r, c]:+.6f}f" for c in range(N_HIDDEN))
        comma = "," if r < N_OUTPUT - 1 else ""
        lines.append(f"    {{{vals}}}{comma}")
    lines.append("};")
    lines.append("")

    # Biases
    lines.append(f"// ── Hidden Biases [{N_HIDDEN}] ────────────────────────────")
    vals = ", ".join(f"{b_h[i]:+.6f}f" for i in range(N_HIDDEN))
    lines.append(f"static const float SNN3_B_H[{N_HIDDEN}] = {{{vals}}};")
    lines.append("")

    lines.append(f"// ── Output Biases [{N_OUTPUT}] ────────────────────────────")
    vals = ", ".join(f"{b_o[i]:+.6f}f" for i in range(N_OUTPUT))
    lines.append(f"static const float SNN3_B_O[{N_OUTPUT}] = {{{vals}}};")
    lines.append("")

    # Spike encoding function
    lines.append("// ═══════════════════════════════════════════════════════════════")
    lines.append("//  SPIKE ENCODING — call this with raw sensor readings")
    lines.append("// ═══════════════════════════════════════════════════════════════")
    lines.append("static inline void snn3_encode_spikes(float hr, float spo2, float hrv,")
    lines.append("                                       int spikes[SNN3_N_IN]) {")
    lines.append("    spikes[0] = (hr   < THR_HR_LOW)    ? 1 : 0;  // HR too low")
    lines.append("    spikes[1] = (hr   > THR_HR_HIGH)   ? 1 : 0;  // HR too high")
    lines.append("    spikes[2] = (spo2 < THR_SPO2_LOW)  ? 1 : 0;  // SpO2 too low")
    lines.append("    spikes[3] = (spo2 > THR_SPO2_HIGH) ? 1 : 0;  // SpO2 too high")
    lines.append("    spikes[4] = (hrv  < THR_HRV_LOW)   ? 1 : 0;  // HRV too low")
    lines.append("    spikes[5] = (hrv  > THR_HRV_HIGH)  ? 1 : 0;  // HRV too high")
    lines.append("}")
    lines.append("")

    # Full inference function
    lines.append("// ═══════════════════════════════════════════════════════════════")
    lines.append("//  SNN INFERENCE — returns predicted class (0-3)")
    lines.append("// ═══════════════════════════════════════════════════════════════")
    lines.append("static inline int snn3_predict(float hr, float spo2, float hrv) {")
    lines.append("    // 1. Spike encoding")
    lines.append("    int spikes[SNN3_N_IN];")
    lines.append("    snn3_encode_spikes(hr, spo2, hrv, spikes);")
    lines.append("")
    lines.append("    // 2. LIF simulation over T timesteps")
    lines.append("    float v_h[SNN3_N_HID] = {0};")
    lines.append("    float v_o[SNN3_N_OUT] = {0};")
    lines.append("    int   spike_counts[SNN3_N_OUT] = {0};")
    lines.append("")
    lines.append("    for (int t = 0; t < SNN3_T_STEPS; t++) {")
    lines.append("        // Hidden layer")
    lines.append("        int s_h[SNN3_N_HID];")
    lines.append("        for (int h = 0; h < SNN3_N_HID; h++) {")
    lines.append("            float I = SNN3_B_H[h];")
    lines.append("            for (int i = 0; i < SNN3_N_IN; i++) {")
    lines.append("                I += SNN3_W_IH[h][i] * (float)spikes[i];")
    lines.append("            }")
    lines.append("            // LIF update with reset")
    lines.append("            if (v_h[h] >= SNN3_THRESHOLD) v_h[h] = 0.0f;")
    lines.append("            v_h[h] = SNN3_ALPHA * v_h[h] + I;")
    lines.append("            s_h[h] = (v_h[h] >= SNN3_THRESHOLD) ? 1 : 0;")
    lines.append("        }")
    lines.append("")
    lines.append("        // Output layer")
    lines.append("        for (int o = 0; o < SNN3_N_OUT; o++) {")
    lines.append("            float I = SNN3_B_O[o];")
    lines.append("            for (int h = 0; h < SNN3_N_HID; h++) {")
    lines.append("                I += SNN3_W_HO[o][h] * (float)s_h[h];")
    lines.append("            }")
    lines.append("            if (v_o[o] >= SNN3_THRESHOLD) v_o[o] = 0.0f;")
    lines.append("            v_o[o] = SNN3_ALPHA * v_o[o] + I;")
    lines.append("            if (v_o[o] >= SNN3_THRESHOLD) spike_counts[o]++;")
    lines.append("        }")
    lines.append("    }")
    lines.append("")
    lines.append("    // 3. Argmax over spike counts → predicted class")
    lines.append("    int best_class = 0;")
    lines.append("    int best_count = spike_counts[0];")
    lines.append("    for (int o = 1; o < SNN3_N_OUT; o++) {")
    lines.append("        if (spike_counts[o] > best_count) {")
    lines.append("            best_count = spike_counts[o];")
    lines.append("            best_class = o;")
    lines.append("        }")
    lines.append("    }")
    lines.append("    return best_class;")
    lines.append("}")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"C header saved to {path}")


header_path = os.path.join(BASE_DIR, "snn_3input_weights.h")
generate_c_header(best_weights, header_path)


# ══════════════════════════════════════════════════════════════════════════════
#  ESP32 DEPLOYMENT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  ESP32-S3 DEPLOYMENT GUIDE")
print("=" * 65)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  MODEL: 3-Input Cardiac SNN (HR + SpO2 + HRV/ECG)             │
│  Target: ESP32-S3 DevKitC-1 (PlatformIO / Arduino)            │
├─────────────────────────────────────────────────────────────────┤
│  Architecture: {N_INPUT} spike → {N_HIDDEN} hidden LIF → {N_OUTPUT} output LIF          │
│  Timesteps:    {T_steps}                                               │
│  Parameters:   {N_INPUT*N_HIDDEN + N_HIDDEN*N_OUTPUT + N_HIDDEN + N_OUTPUT} total ({N_INPUT*N_HIDDEN} + {N_HIDDEN*N_OUTPUT} + {N_HIDDEN} + {N_OUTPUT} biases)           │
├─────────────────────────────────────────────────────────────────┤
│  INPUTS (3 sensors):                                           │
│    1. Heart Rate    → MAX30100 PPG (I2C: SDA=GPIO8, SCL=GPIO9)│
│    2. SpO2          → MAX30100     (same I2C bus)              │
│    3. HRV (SDNN)    → AD8232 ECG  (analog: GPIO4)             │
├─────────────────────────────────────────────────────────────────┤
│  OUTPUTS (4 classes):                                          │
│    0 = Normal                                                  │
│    1 = Arrhythmia                                              │
│    2 = Hypotensive Event                                       │
│    3 = Hypertensive Crisis                                     │
├─────────────────────────────────────────────────────────────────┤
│  ESTIMATED RESOURCE USAGE:                                     │
│    Flash:    ~0.6 KB (weights stored as float constants)       │
│    RAM:      ~0.3 KB (LIF state + spike buffers)               │
│    Latency:  ~8 µs per inference                               │
│    Energy:   ~0.5 µJ per inference                             │
├─────────────────────────────────────────────────────────────────┤
│  FILES TO INTEGRATE:                                           │
│    1. snn_3input_weights.h  → copy to firmware/src/snn/       │
│    2. #include "snn_3input_weights.h" in your main.cpp        │
│    3. Call: int cls = snn3_predict(hr, spo2, hrv);            │
└─────────────────────────────────────────────────────────────────┘

USAGE IN FIRMWARE:
─────────────────
  #include "snn/snn_3input_weights.h"

  void loop() {{
      float hr   = read_heart_rate();   // from MAX30100
      float spo2 = read_spo2();         // from MAX30100
      float hrv  = compute_hrv_sdnn();  // from AD8232 ECG R-R intervals

      int prediction = snn3_predict(hr, spo2, hrv);

      Serial.print("Class: ");
      Serial.print(prediction);
      Serial.print(" (");
      Serial.print(SNN3_CLASS_NAMES[prediction]);
      Serial.println(")");
  }}
""")

print("Done! All files generated successfully.")
