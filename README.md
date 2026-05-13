# SoH Estimation for Sodium-Ion Batteries Under Partial Charging Conditions Using Physical Capacity Features and Lightweight Gradient Boosting 

Fork of [BatteryLife (KDD 2025)](https://github.com/Ruifeng-Tan/BatteryLife), extended with a physical battery characteristics based SoH estimation pipeline for Na-ion batteries using incremental capacity analysis(ICA) features and XGBoost.

---

## Setup

```bash
pip install -r requirements.txt
pip install git+https://github.com/microsoft/BatteryML.git --no-deps
pip install evaluate reformer_pytorch denseweight deepspeed addict fire
```

Download the NA-ion dataset:

```bash
huggingface-cli login
hf download Battery-Life/BatteryLife_Processed \
  --repo-type dataset \
  --include "NA-ion/*" "Life labels/*" "seen_unseen_labels/*" \
  --local-dir ./dataset
```

---

## Reproducing Results

**Baseline models (DLinear / CPMLP):**
```bash
python run_main.py --dataset NAion --model [CPMLP] \
  --early_cycle_threshold [100|50] --root_path ./dataset
```

**ICA+XGBoost (our contribution):**
```bash
python run_ica_xgboost.py --root_path ./dataset/NA-ion
```

---

## Authors
Osman Huseynov · Emir Selman Karaatlı  
MYZ307E — Istanbul Technical University
Spring 2026
