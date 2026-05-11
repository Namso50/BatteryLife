# ICA + XGBoost SOH Estimator
# Extracts 6 physics-based ICA features per cycle and trains an XGBoost regressor
# to estimate State of Health (SOH) from partial charging data.
#
# Usage:
#   python run_ica_xgboost.py --root_path ./dataset/NA-ion
#   python run_ica_xgboost.py --root_path ./dataset/NA-ion --partial_lower 0.0 --partial_upper 1.0
#
# Key arguments:
#   --root_path       Path to NA-ion .pkl files (default: ./dataset/NA-ion)
#   --partial_lower   Lower SOC bound, e.g. 0.20 for 20% (default: 0.20)
#   --partial_upper   Upper SOC bound, e.g. 0.80 for 80% (default: 0.80)
#   --n_estimators    Number of XGBoost trees (default: 500)
#   --n_blind         Number of batteries held out for blind test (default: 5)
#   --seed            Random seed for reproducibility (default: 42)

import os
import argparse
import random
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description='ICA + XGBoost SOH Estimator')
parser.add_argument('--root_path',     type=str,   default='./dataset/NA-ion')
parser.add_argument('--partial_lower', type=float, default=0.20,
                    help='Lower SOC bound for partial charging window (default: 0.20)')
parser.add_argument('--partial_upper', type=float, default=0.80,
                    help='Upper SOC bound for partial charging window (default: 0.80)')
parser.add_argument('--n_estimators',  type=int,   default=500)
parser.add_argument('--learning_rate', type=float, default=0.03)
parser.add_argument('--max_depth',     type=int,   default=7)
parser.add_argument('--n_blind',       type=int,   default=5,
                    help='Number of batteries held out for blind test')
parser.add_argument('--seed',          type=int,   default=42)
args = parser.parse_args()

# ============================================================
# Feature extraction
# ============================================================
def extract_features(file_path, partial_lower=None, partial_upper=None):
    try:
        data   = pd.read_pickle(file_path)
        cycles = data['cycle_data']
        results = []
        for c_idx in range(len(cycles)):
            c_df  = pd.DataFrame(cycles[c_idx])
            v_col = 'voltage_in_V'
            i_col = 'current_in_A'

            dis_mask = c_df[i_col] < -1.0
            df_dis   = c_df[dis_mask]
            if df_dis.empty:
                continue

            if partial_lower is None:
                ch_mask = c_df[i_col] > 1.0
            else:
                full_ch = c_df[c_df[i_col] > 1.0]
                if full_ch.empty:
                    continue
                v_min  = full_ch[v_col].min()
                v_max  = full_ch[v_col].max()
                v_low  = v_min + (v_max - v_min) * partial_lower
                v_high = v_min + (v_max - v_min) * partial_upper
                ch_mask = (c_df[i_col] > 1.0) & \
                          (c_df[v_col] >= v_low) & \
                          (c_df[v_col] <= v_high)

            df_ch = c_df[ch_mask].copy().reset_index(drop=True)
            if df_ch.empty:
                continue

            time_s = df_ch['time_in_s'].values
            dt     = np.diff(time_s, prepend=time_s[0])
            dt[dt <= 0] = 1.0
            df_ch['Cap'] = np.cumsum(df_ch[i_col].values * dt / 3600)

            w = 51 if len(df_ch) > 51 else \
                (len(df_ch)-1 if len(df_ch)%2==0 else 3)
            if w <= 3:
                continue

            df_ch['V_s']   = savgol_filter(df_ch[v_col], window_length=w, polyorder=3)
            dQ             = np.gradient(df_ch['Cap'])
            dV             = np.gradient(df_ch['V_s'])
            df_ch['dQdV']  = dQ / (dV + 1e-6)

            m1 = (df_ch['V_s'] >= 2.7) & (df_ch['V_s'] <= 3.2)
            m2 = (df_ch['V_s'] >= 3.4) & (df_ch['V_s'] <= 3.9)
            if df_ch[m1].empty or df_ch[m2].empty:
                continue

            idx1        = df_ch.loc[m1, 'dQdV'].idxmax()
            idx2        = df_ch.loc[m2, 'dQdV'].idxmax()
            v1, dq1     = df_ch.loc[idx1, ['V_s', 'dQdV']]
            v2, dq2     = df_ch.loc[idx2, ['V_s', 'dQdV']]
            m_val       = (df_ch['V_s'] > v1) & (df_ch['V_s'] < v2)
            if df_ch[m_val].empty:
                continue
            idx_v       = df_ch.loc[m_val, 'dQdV'].idxmin()
            v_val, dq_val = df_ch.loc[idx_v, ['V_s', 'dQdV']]

            true_cap = (abs(df_dis[i_col].values) *
                        np.diff(df_dis['time_in_s'].values,
                                prepend=df_dis['time_in_s'].values[0]) / 3600).sum()

            results.append({
                'P1_V': v1,    'P1_dQ':  dq1,
                'Val_V': v_val,'Val_dQ': dq_val,
                'P2_V': v2,    'P2_dQ':  dq2,
                'cap':  true_cap
            })
        return results
    except Exception:
        return []

# ============================================================
# Data split
# ============================================================
random.seed(args.seed)
all_files   = sorted([f for f in os.listdir(args.root_path) if f.endswith('.pkl')])
blind_files = random.sample(all_files, args.n_blind)
train_files = [f for f in all_files if f not in blind_files]

feature_cols = ['P1_V','P1_dQ','Val_V','Val_dQ','P2_V','P2_dQ']

window_str = (f"{int(args.partial_lower*100)}–{int(args.partial_upper*100)}% SOC"
              if args.partial_lower is not None else "0–100% (full)")

print(f"\n{'='*55}")
print(f"ICA + XGBoost SOH Estimator")
print(f"{'='*55}")
print(f"Dataset     : {args.root_path}")
print(f"Window      : {window_str}")
print(f"Train files : {len(train_files)} | Blind test: {len(blind_files)}")
print(f"XGBoost     : n_estimators={args.n_estimators}, "
      f"lr={args.learning_rate}, max_depth={args.max_depth}")
print(f"{'='*55}\n")

# ============================================================
# Train
# ============================================================
print("Extracting features...")
master = []
for f in train_files:
    master.extend(extract_features(
        os.path.join(args.root_path, f),
        args.partial_lower, args.partial_upper))

df_train = pd.DataFrame(master).dropna()
print(f"Training cycles: {len(df_train)}\n")

model = xgb.XGBRegressor(
    n_estimators=args.n_estimators,
    learning_rate=args.learning_rate,
    max_depth=args.max_depth,
    subsample=0.8,
    random_state=args.seed,
    verbosity=0)
model.fit(df_train[feature_cols], df_train['cap'])

# ============================================================
# Blind test
# ============================================================
print("Blind test results:")
all_true, all_pred = [], []
for f in blind_files:
    rows  = extract_features(os.path.join(args.root_path, f),
                             args.partial_lower, args.partial_upper)
    df_te = pd.DataFrame(rows).dropna(subset=feature_cols+['cap'])
    if df_te.empty:
        continue
    y_true = df_te['cap'].values
    y_pred = model.predict(df_te[feature_cols])
    mape   = mean_absolute_percentage_error(y_true, y_pred) * 100
    acc15  = np.mean(np.abs((y_true-y_pred)/y_true) <= 0.15) * 100
    acc1   = np.mean(np.abs((y_true-y_pred)/y_true) <= 0.01) * 100
    all_true.extend(y_true)
    all_pred.extend(y_pred)
    print(f"  {f[-25:]:>27}  MAPE:{mape:6.2f}%  "
          f"±15%:{acc15:6.1f}%  ±1%:{acc1:6.1f}%")

all_true = np.array(all_true)
all_pred = np.array(all_pred)
mape_total  = mean_absolute_percentage_error(all_true, all_pred) * 100
acc15_total = np.mean(np.abs((all_true-all_pred)/all_true) <= 0.15) * 100
acc1_total  = np.mean(np.abs((all_true-all_pred)/all_true) <= 0.01) * 100

print(f"\n{'='*55}")
print(f"Overall — MAPE: {mape_total:.3f}%  "
      f"±15% Acc: {acc15_total:.1f}%  ±1% Acc: {acc1_total:.1f}%")
print(f"{'='*55}")