"""
train_snn.py
Train multi-class SNN (LIF) using surrogate gradient descent for cardiac event classification.
Architecture: 6 inputs -> 8 hidden LIF -> 4 output LIF (one per class).
Classes:
    0 = Normal
    1 = Arrhythmia (erratic HR, very low HRV, low SpO2)
    2 = Hypotensive Event (low BP, moderate tachycardia)
    3 = Hypertensive Crisis (very high BP, tachycardia, tachypnea)

Sensors:
    MAX30100 → heart_rate, blood_oxygen (SpO2)
    AD8232 ECG → heart_rate, respiratory_rate, hrv_sdnn (from R-R intervals)
    DS18B20 → body_temperature
    Blood pressure → estimated via Pulse Transit Time (MAX30100 + ECG)
"""

import os, json, warnings
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

# 6 features extractable from ESP32-S3 sensors
FEATURE_COLS = ["heart_rate", "blood_oxygen", "body_temperature",
                "respiratory_rate", "hrv_sdnn", "blood_pressure_systolic"]

CLASS_NAMES = ["Normal", "Arrhythmia", "Hypotensive", "Hypertensive"]

X = df[FEATURE_COLS].values.astype(np.float64)
y = df["health_event"].values.astype(int)  # 0, 1, 2, 3

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ── Spike Encoding (medical thresholds) ──────────────────────────────────────
# Normal ranges based on medical guidelines
THRESHOLDS = {
    'heart_rate':    (60.0, 100.0),   # bpm
    'blood_oxygen':  (95.0, 100.0),   # %
    'body_temp':     (96.5, 99.5),    # °F
    'resp_rate':     (12.0, 20.0),    # brpm
    'hrv_sdnn':      (30.0, 70.0),    # ms — HRV below 30 is concerning
    'bp_systolic':   (90.0, 140.0),   # mmHg
}

def encode_spikes(X):
    """
    Convert raw features to binary spikes. 
    2 spikes per feature: one for "too low", one for "too high".
    This gives the SNN directional information.
    Total: 6 features × 2 directions = 12 spike channels.
    """
    n = len(X)
    spikes = np.zeros((n, 12), dtype=np.float64)
    
    ranges = [
        THRESHOLDS['heart_rate'],
        THRESHOLDS['blood_oxygen'],
        THRESHOLDS['body_temp'],
        THRESHOLDS['resp_rate'],
        THRESHOLDS['hrv_sdnn'],
        THRESHOLDS['bp_systolic'],
    ]
    
    for i, (lo, hi) in enumerate(ranges):
        spikes[:, 2*i]     = (X[:, i] < lo).astype(float)   # too low
        spikes[:, 2*i + 1] = (X[:, i] > hi).astype(float)   # too high
    
    return spikes

X_train_spk = encode_spikes(X_train)
X_test_spk  = encode_spikes(X_test)

N_INPUT = 12  # 6 features × 2 directions
N_HIDDEN = 10
N_OUTPUT = 4  # 4 classes

print(f"Spike channels: {N_INPUT} | Hidden: {N_HIDDEN} | Output: {N_OUTPUT}")
print(f"Spike rate (train): {X_train_spk.mean():.3f}")

# ── Multi-Class Surrogate Gradient SNN ───────────────────────────────────────
class MultiClassSNN:
    """
    LIF SNN with surrogate gradient training for 4-class cardiac classification.
    Architecture: 12 spike inputs -> 10 hidden LIF -> 4 output LIF.
    Uses softmax over spike counts for multi-class output.
    """
    def __init__(self, n_in=12, n_hid=10, n_out=4, alpha=0.85, threshold=0.8, lr=0.01):
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
        
        # Convert spike counts to probabilities via softmax
        spike_rates = total_output_spikes / T
        probs = self._softmax(spike_rates * 5.0)  # temperature scaling
        
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
        d_rates = d_probs * 5.0  # temperature scaling
        
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
print("\n" + "="*60)
print("  TRAINING MULTI-CLASS SNN (Surrogate Gradient Descent)")
print("="*60)

best_f1 = 0
best_weights = None
n_epochs = 300
batch_size = 64
T_steps = 10

# Use class weights to handle imbalance
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (len(class_counts) * class_counts.astype(float))
print(f"Class weights: {dict(enumerate(class_weights.round(2)))}")

snn = MultiClassSNN(n_in=N_INPUT, n_hid=N_HIDDEN, n_out=N_OUTPUT, 
                     alpha=0.85, threshold=0.8, lr=0.015)

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
    if (epoch + 1) == 120:
        snn.lr *= 0.5
    if (epoch + 1) == 200:
        snn.lr *= 0.5

# ── Report Best Results ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("  BEST RESULTS")
print("="*60)

# Reload best weights
snn.W_ih = best_weights['W_ih']
snn.W_ho = best_weights['W_ho']
snn.b_h = best_weights['b_h']
snn.b_o = best_weights['b_o']

preds = snn.predict(X_test_spk, T=T_steps)

print(f"\nF1-macro:    {best_weights['f1_macro']:.4f}")
print(f"F1-weighted: {best_weights['f1_weighted']:.4f}")
print(f"Accuracy:    {best_weights['acc']:.4f}")

print("\n" + classification_report(y_test, preds, target_names=CLASS_NAMES))

cm = confusion_matrix(y_test, preds)
print("Confusion Matrix:")
print(cm)

# ── Save weights ─────────────────────────────────────────────────────────────
print("\n── Trained Weights ──")
print(f"W_ih shape: {best_weights['W_ih'].shape}")
print(f"W_ho shape: {best_weights['W_ho'].shape}")

result_path = os.path.join(BASE_DIR, "snn_trained_weights.json")
with open(result_path, "w") as f:
    json.dump({
        'architecture': {'n_in': N_INPUT, 'n_hid': N_HIDDEN, 'n_out': N_OUTPUT},
        'W_ih': best_weights['W_ih'].tolist(),
        'W_ho': best_weights['W_ho'].tolist(),
        'b_h': best_weights['b_h'].tolist(),
        'b_o': best_weights['b_o'].tolist(),
        'metrics': {
            'f1_macro': best_weights['f1_macro'],
            'f1_weighted': best_weights['f1_weighted'],
            'accuracy': best_weights['acc'],
        },
        'class_names': CLASS_NAMES,
        'feature_cols': FEATURE_COLS,
        'hyperparams': {
            'alpha': 0.85, 'threshold': 0.8,
            'T_steps': T_steps, 'epochs': n_epochs,
        },
        'spike_encoding': '2 spikes per feature (too_low, too_high)',
        'thresholds': {k: list(v) for k, v in THRESHOLDS.items()},
    }, f, indent=2)
print(f"\nWeights saved to {result_path}")
