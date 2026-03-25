import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

print("="*60)
print("DeepPredict Model Optimization Verification")
print("="*60)

results = {}

# Load test data
df = pd.read_csv('test_data/ETTh1.csv')
date_cols = [c for c in df.columns if 'date' in c.lower()]
df = df.drop(columns=date_cols, errors='ignore').dropna()

print(f"\nData: {df.shape[0]} samples, {df.shape[1]-1} features")

y = df['OT'].values.astype(np.float32)
X = df.drop(columns=['OT']).values.astype(np.float32)

# Test 1: Small dataset (first 200 samples) - CNN1D
print("\n" + "="*60)
print("TEST 1: CNN1D on SMALL dataset (200 samples)")
print("="*60)
from src.models.cnn1d_model import CNN1DPredictorV4

X_small = X[:200]
y_small = y[:200]

for run in range(3):
    model = CNN1DPredictorV4()
    ok, msg = model.train(
        X_small, y_small,
        seq_len=24, pred_len=12,
        hidden_channels=32,
        epochs=30, batch_size=8,
        test_size=0.2
    )
    if ok:
        print(f"  Run{run+1}: R2={model.metrics['R2']:.4f} RMSE={model.metrics['RMSE']:.4f}")
        results.setdefault('CNN1D_small', []).append(model.metrics['R2'])
    else:
        print(f"  Run{run+1}: FAILED - {msg[:100]}")

# Test 2: Small dataset - LSTM
print("\n" + "="*60)
print("TEST 2: LSTM on SMALL dataset (200 samples)")
print("="*60)
from src.models.lstm_model import LSTMPredictor

for run in range(3):
    model = LSTMPredictor()
    ok, msg = model.train(
        X_small, y_small,
        hidden_size=32, num_layers=1,
        seq_len=10, epochs=30, batch_size=16,
        test_size=0.2
    )
    if ok:
        print(f"  Run{run+1}: R2={model.metrics['R2']:.4f} RMSE={model.metrics['RMSE']:.4f}")
        results.setdefault('LSTM_small', []).append(model.metrics['R2'])
    else:
        print(f"  Run{run+1}: FAILED - {msg[:100]}")

# Test 3: Small dataset - PatchTST
print("\n" + "="*60)
print("TEST 3: PatchTST on SMALL dataset (200 samples)")
print("="*60)
from src.models.patchtst_model import PatchTSTPredictor

for run in range(3):
    model = PatchTSTPredictor()
    ok, msg = model.train(
        X_small, y_small,
        seq_len=24, pred_len=12,
        d_model=32, n_layers=1, n_heads=2,
        epochs=30, batch_size=8,
        test_size=0.2
    )
    if ok:
        print(f"  Run{run+1}: R2={model.metrics['R2']:.4f} RMSE={model.metrics['RMSE']:.4f}")
        results.setdefault('PatchTST_small', []).append(model.metrics['R2'])
    else:
        print(f"  Run{run+1}: FAILED - {msg[:100]}")

# Test 4: Full dataset - CNN1D (larger data)
print("\n" + "="*60)
print("TEST 4: CNN1D on FULL dataset (all samples)")
print("="*60)
for run in range(3):
    model = CNN1DPredictorV4()
    ok, msg = model.train(
        X, y,
        seq_len=96, pred_len=48,
        hidden_channels=64,
        epochs=30, batch_size=32,
        test_size=0.2
    )
    if ok:
        print(f"  Run{run+1}: R2={model.metrics['R2']:.4f} RMSE={model.metrics['RMSE']:.4f}")
        results.setdefault('CNN1D_full', []).append(model.metrics['R2'])
    else:
        print(f"  Run{run+1}: FAILED - {msg[:100]}")

# Test 5: Full dataset - LSTM
print("\n" + "="*60)
print("TEST 5: LSTM on FULL dataset (all samples)")
print("="*60)
for run in range(3):
    model = LSTMPredictor()
    ok, msg = model.train(
        X, y,
        hidden_size=64, num_layers=2,
        seq_len=24, epochs=50, batch_size=32,
        test_size=0.2
    )
    if ok:
        print(f"  Run{run+1}: R2={model.metrics['R2']:.4f} RMSE={model.metrics['RMSE']:.4f}")
        results.setdefault('LSTM_full', []).append(model.metrics['R2'])
    else:
        print(f"  Run{run+1}: FAILED - {msg[:100]}")

# Test 6: Full dataset - PatchTST
print("\n" + "="*60)
print("TEST 6: PatchTST on FULL dataset (all samples)")
print("="*60)
for run in range(3):
    model = PatchTSTPredictor()
    ok, msg = model.train(
        X, y,
        seq_len=96, pred_len=48,
        d_model=64, n_layers=2, n_heads=4,
        epochs=30, batch_size=16,
        test_size=0.2
    )
    if ok:
        print(f"  Run{run+1}: R2={model.metrics['R2']:.4f} RMSE={model.metrics['RMSE']:.4f}")
        results.setdefault('PatchTST_full', []).append(model.metrics['R2'])
    else:
        print(f"  Run{run+1}: FAILED - {msg[:100]}")

# Summary
print("\n" + "="*60)
print("FINAL SUMMARY - R² Scores (higher is better, >0 is good)")
print("="*60)
for name, vals in sorted(results.items()):
    avg = np.mean(vals)
    std = np.std(vals)
    best = max(vals)
    worst = min(vals)
    status = "✓" if avg > 0 else "✗"
    print(f"  {status} {name:20s}: avg={avg:+.4f}  best={best:+.4f}  worst={worst:+.4f}  std={std:.4f}")

print("\n" + "="*60)
print("Optimization verification complete!")
print("="*60)
