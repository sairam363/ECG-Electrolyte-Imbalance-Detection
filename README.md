# ECG Electrolyte Imbalance Detection using Deep Learning

## Overview
This project focuses on detecting electrolyte imbalances such as Hyperkalemia, Hypokalemia, Hypercalcemia, and Hypocalcemia from ECG waveforms using Deep Learning techniques. The model is trained using the PTB-XL ECG dataset and aims to assist in early medical diagnosis through automated ECG analysis.

The project includes:
- ECG signal preprocessing
- Deep Learning model training
- Electrolyte imbalance classification
- Streamlit-based web interface for prediction

---

## Features
- Detects multiple electrolyte imbalance conditions from ECG data
- Lightweight Deep Learning architecture for ECG analysis
- Automated preprocessing pipeline
- Streamlit web application for real-time predictions
- Scalable and modular project structure

---

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit

---

## Dataset
Dataset used:
- PTB-XL ECG Dataset from PhysioNet

The dataset contains:
- Large-scale 12-lead ECG recordings
- Diagnostic statements
- Metadata and waveform signals

Dataset Link:
https://physionet.org/content/ptb-xl/1.0.3/

---

## Project Structure

ECG-Electrolyte-Imbalance-Detection/
│
├── app.py
├── main.py
├── debug.py
├── file.py
├── model.pkl
├── requirements.txt
├── README.md
└── screenshots/

---

## Model Workflow
1. Load ECG waveform data
2. Preprocess ECG signals
3. Extract relevant features
4. Train Deep Learning model
5. Predict electrolyte imbalance category
6. Display results through Streamlit UI

---

## Installation

### Clone Repository
```bash
git clone https://github.com/sairam363/ECG-Electrolyte-Imbalance-Detection.git
cd ECG-Electrolyte-Imbalance-Detection
