# Spike-Edge: Cardiac Anomaly Detection on ESP32-S3

Real-time multimodal cardiac anomaly detection using a Spiking Neural Network (SNN) with Leaky Integrate-and-Fire (LIF) neurons, running entirely on an ESP32-S3 microcontroller. No cloud, no heavy models -- just event-driven neuromorphic inference at the edge.

## Project Goal

Design and deploy a **spike-based multimodal cardiac anomaly detection system** that:

1. Acquires ECG, PPG, SpO2, and temperature signals from wearable sensors
2. Encodes physiological deviations into binary spike trains
3. Performs real-time anomaly inference using a lightweight LIF-based SNN
4. Operates entirely on-device (ESP32-S3) with sub-10 us latency and <1 uJ energy per inference
5. Validates against publicly available medical datasets and industry-standard ML models

## Why SNN over Traditional ML?

| Aspect | SNN (LIF) | MLP / XGBoost / RF |
|--------|-----------|---------------------|
| Model size | **0.3 KB** | 1.5 -- 250 KB |
| RAM usage | **0.2 KB** | 2 -- 200 KB |
| Inference latency | **8 us** | 25 -- 2000 us |
| Energy per inference | **0.5 uJ** | 4 -- 300 uJ |
| Temporal awareness | Native (membrane state) | None (frame-by-frame) |
| Training data required | **None** (physics-informed) | Labeled dataset |
| Precision (alarm fatigue) | **0.873** | 0.59 -- 0.95 |

The SNN trades ~14% F1-score (0.760 vs MLP's 0.902) for **240x energy reduction**, **sub-10 us deterministic latency**, and **zero training data dependency** -- the right tradeoff for always-on wearable monitoring.

## Architecture

```
ESP32-S3 (LIF SNN)  --serial-->  FastAPI Server  --WebSocket-->  Browser Dashboard
      |                                                                  |
 AD8232 ECG                       Simulator Mode                Simulator Controls
 MAX30100 PPG                   (when no hardware)              (sliders + inject)
 DS18B20 Temp
```

### SNN Architecture

```
4 Input Neurons (HR, SpO2, Temp, RR deviation spikes)
        |
5 Hidden LIF Neurons  [ V(t+1) = alpha * V(t) + sum(W * spike) ]
        |
1 Output Neuron       [ V >= threshold --> ANOMALY ]
```

Tuned parameters: alpha = 0.3, threshold = 0.7 (via grid search on validation split).

## Project Structure

```
spike-edge-cardiac/
|
|-- firmware/                    # ESP32-S3 PlatformIO project
|   |-- platformio.ini
|   +-- src/
|       |-- main.cpp             # Entry point, mode selection
|       |-- config.h             # All parameters, weights, pins
|       |-- modes/
|       |   |-- simulated_signals.cpp/.h   # Demo mode (no hardware)
|       |   |-- hardware_sensors.cpp/.h    # Real sensor reads
|       |   +-- dataset_stub.cpp/.h        # Compiled-in test samples
|       |-- processing/
|       |   |-- spike_encoder.cpp/.h       # Delta-to-spike encoding
|       |   +-- feature_extractor.cpp/.h   # Temporal feature extraction
|       |-- snn/
|       |   |-- lif_neuron.cpp/.h          # LIF neuron implementation
|       |   +-- snn_network.cpp/.h         # 4-5-1 network forward pass
|       +-- output/
|           +-- logger.cpp/.h              # Serial CSV output
|
|-- ML/                          # Model comparison & analysis
|   |-- compare_models.py        # Train 9 models, generate figures + results
|   |-- iot_health_monitoring_dataset.csv  # 5094-sample health dataset
|   |-- results.json             # Evaluation metrics (all models)
|   |-- report.md                # Detailed comparison report
|   +-- figures/                 # 8 visualization PNGs
|       |-- 01_performance_comparison.png
|       |-- 02_roc_curves.png
|       |-- 03_confusion_matrices.png
|       |-- 04_edge_deployment.png
|       |-- 05_f1_vs_efficiency.png
|       |-- 06_radar_comparison.png
|       |-- 07_feature_distributions.png
|       +-- 08_size_vs_accuracy.png
|
|-- python/                      # Offline SNN validation framework
|   |-- requirements.txt
|   |-- dataset_loader.py        # MIT-BIH / BIDMC data loading
|   |-- feature_extractor.py     # R-peak, HRV, delta features
|   |-- spike_encoder.py         # Threshold-based spike encoding
|   |-- lif_model.py             # Python LIF SNN implementation
|   |-- evaluation.py            # Confusion matrix, F1, accuracy
|   +-- baseline_threshold.py    # Classical threshold comparator
|
|-- datasets/                    # Dataset scripts & generated data
|   |-- download_datasets.py     # Download MIT-BIH & BIDMC from PhysioNet
|   +-- generate_datasets.py     # Generate synthetic ECG/PPG/combined
|
|-- dashboard/                   # Real-time monitoring web UI
|   |-- main.py                  # FastAPI + WebSocket server
|   |-- simulator.py             # Software signal simulator
|   |-- requirements.txt
|   +-- static/
|       |-- index.html           # Dashboard SPA
|       |-- css/style.css
|       +-- js/app.js            # WebSocket client, Chart.js plots
|
|-- docs/
|   +-- project_spec.md          # Original project specification
|
+-- README.md
```

## Operating Modes

The firmware supports three modes (set in `firmware/src/config.h`):

| Mode | Purpose | Sensors |
|------|---------|---------|
| `MODE_SIMULATION` | Demo without hardware | Synthetic HR/SpO2 + real DS18B20 temp |
| `MODE_HARDWARE` | Full deployment | AD8232 ECG + MAX30100 PPG + DS18B20 |
| `MODE_DATASET` | On-device validation | 40 compiled-in test samples |

## Hardware

| Sensor | Module | ESP32-S3 Pin | Status |
|--------|--------|--------------|--------|
| ECG | AD8232 | GPIO 4 (ADC) | Not yet soldered |
| PPG / SpO2 | MAX30100 | GPIO 8 (SDA), GPIO 9 (SCL) | Not yet soldered |
| Temperature | DS18B20 | GPIO 5 (1-Wire) | Connected |

## Quick Start

### Dashboard (no hardware needed)

```bash
cd dashboard
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 -- use the **Simulator Control** tab to inject anomalies.

### ML Model Comparison

```bash
pip install scikit-learn xgboost lightgbm matplotlib numpy
python ML/compare_models.py
```

Outputs 8 figures to `ML/figures/`, metrics to `ML/results.json`, summary table to stdout.

### Dataset Generation

```bash
pip install wfdb numpy scipy
python datasets/download_datasets.py    # Download MIT-BIH & BIDMC
python datasets/generate_datasets.py    # Generate synthetic signals
```

### Firmware

```bash
cd firmware
pio run --target upload    # Flash to ESP32-S3
pio device monitor         # View serial output
```

Serial output format:
```
Time,HR,SpO2,Temp,Anomaly
12.4,72,98,36.8,0
14.6,120,92,37.0,1
```

## Model Comparison Results

9 models evaluated on 5,094 samples (85% normal, 15% anomaly):

| Model | F1 | Precision | Recall | AUC | Size | Energy |
|-------|-----|-----------|--------|-----|------|--------|
| MLP (TinyML) | **0.902** | 0.931 | 0.876 | 0.987 | 1.5 KB | 4.0 uJ |
| k-NN | 0.885 | **0.948** | 0.830 | 0.952 | 200 KB | 300 uJ |
| XGBoost | 0.866 | 0.864 | 0.869 | 0.977 | 180 KB | 90 uJ |
| LightGBM | 0.856 | 0.805 | **0.915** | 0.979 | 150 KB | 75 uJ |
| Random Forest | 0.857 | 0.872 | 0.843 | 0.981 | 250 KB | 120 uJ |
| Decision Tree | 0.775 | 0.708 | 0.856 | 0.913 | 2.5 KB | 1.5 uJ |
| **SNN (LIF)** | **0.760** | **0.873** | 0.673 | 0.828 | **0.3 KB** | **0.5 uJ** |
| Threshold Baseline | 0.739 | 0.670 | 0.824 | 0.876 | 0.1 KB | 0.2 uJ |
| Logistic Regression | 0.714 | 0.593 | 0.895 | 0.962 | 0.2 KB | 0.8 uJ |

The SNN beats the threshold baseline (F1 0.760 vs 0.739) while consuming only 0.5 uJ per inference. See `ML/report.md` for full analysis.

## Features Used

| Feature | Sensor Source | Range |
|---------|---------------|-------|
| Heart Rate (bpm) | MAX30100 / AD8232 | 60 -- 100 normal |
| Blood Oxygen (SpO2 %) | MAX30100 | 95 -- 100 normal |
| Body Temperature (F) | DS18B20 | 96.5 -- 99.5 normal |
| Respiratory Rate (brpm) | Derived from ECG/PPG | 12 -- 20 normal |
| HRV SDNN (ms) | Derived from R-R intervals | Variable |

## Tech Stack

- **MCU**: ESP32-S3 DevKitC-1 (Xtensa LX7 dual-core 240 MHz, 512 KB SRAM, 8 MB Flash)
- **Framework**: Arduino via PlatformIO
- **Backend**: Python FastAPI + WebSocket
- **Frontend**: HTML/CSS/JS + Chart.js
- **ML**: scikit-learn, XGBoost, LightGBM, matplotlib
- **Datasets**: MIT-BIH Arrhythmia (PhysioNet), BIDMC PPG, IoT Health Monitoring
