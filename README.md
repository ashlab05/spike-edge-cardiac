# Spike-Edge Cardiac Anomaly Detection

Real-time multimodal cardiac anomaly detection using a Spiking Neural Network (SNN) on ESP32-S3. Combines ECG, PPG, SpO₂, and temperature signals with event-driven neuromorphic inference.

## Architecture

```
ESP32-S3 (LIF SNN)  ──serial──►  FastAPI Server  ──WebSocket──►  Browser Dashboard
      ↕                                                                      ↕
 AD8232 ECG                       Simulator Mode                    Simulator Controls
 MAX30100 PPG                    (when no hardware)                 (sliders + inject)
 DS18B20 Temp
```

## Quick Demo (No Hardware Required)

Run the dashboard with the built-in signal simulator:

```bash
cd dashboard
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

**Live Monitor tab** — real-time scrolling charts for HR, SpO₂, and Temperature with anomaly detection.

**Simulator Control tab** — drag sliders to override signal values, click **Inject Anomaly** for a 5-second burst (HR→120, SpO₂→92).

## Download Datasets

```bash
cd datasets
pip install wfdb
python download_datasets.py
```

Downloads:
- **MIT-BIH Arrhythmia Database** (PhysioNet) → `datasets/mit-bih/`
- **BIDMC PPG Dataset** (PhysioNet) → `datasets/bidmc/`

## Python Validation Framework

```bash
cd python
pip install -r requirements.txt
python evaluation.py
```

Runs SNN vs threshold-based baseline on MIT-BIH records and prints confusion matrix, accuracy, F1-score.

## Hardware (After Soldering)

| Sensor | Module | ESP32-S3 Pin |
|--------|--------|--------------|
| ECG | AD8232 | GPIO 4 (ADC) |
| PPG/SpO₂ | MAX30100 | GPIO 8 (SDA), GPIO 9 (SCL) |
| Temperature | DS18B20 | GPIO 5 (1-Wire) |

Flash firmware with PlatformIO:
```bash
cd firmware
pio run --target upload
```

## SNN Architecture

```
4 Input Neurons (HR spike, SpO₂ spike, Temp spike, RR spike)
        ↓
5 Hidden LIF Neurons  (V(t+1) = 0.9·V(t) + ΣW·spike)
        ↓
1 Output Neuron  (V ≥ 1.0 → ANOMALY)
```

## Project Structure

```
spike-edge-cardiac/
├── dashboard/          # FastAPI server + WebSocket + frontend
├── python/             # Dataset validation + SNN training
├── firmware/           # ESP32-S3 PlatformIO project
└── datasets/           # Dataset download scripts
```

## Signal Output Format (Serial)

```
Time,HR,SpO2,Temp,Anomaly
12.4,72,98,36.8,0
14.6,120,92,37.0,1
```
