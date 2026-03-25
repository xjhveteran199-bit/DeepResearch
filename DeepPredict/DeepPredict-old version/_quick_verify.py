import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

print("="*60)
print("Quick Model Verification Test")
print("="*60)

# Load test data
df = pd.read_csv('test_data/ETTh1.csv')
date_cols = [c for c in df.columns if 'date' in c.lower()]
df = df.drop(columns=date_cols, errors='ignore').dropna()

print(f"\nData: {df.shape[0]} samples, {df.shape[1]-1} features")

y = df['OT'].values.astype(np.float32)
X = df.drop(columns=['OT']).values.astype(np.float32)

results = {}

# Test 1: Small dataset - CNN1D
print("\n--- CNN1D (small data 200 samples) ---")
from src.models.cnn1d_model import CNN1DPredictorV4

for run in range(2):
    model = CNN1DPredictorV4()
    ok, msg = model.train(X[:200], y[:200], seq_len=24, pred_len=12, 
                          hidden_channels=32, epochs=20, batch_size=8, test_size=0.2)
    if ok:
        print(f"  Run{run+1}: R2={model.metrics['R2']:.4f}")
        results.setdefault('CNN1D_small', []).append(model.metrics['R2'])
    else:
        print(f"  Run{run+1}: FAILED")

# Test 2: Small dataset - LSTM
print("\n--- LSTM (small data 200 samples) ---")
from src.models.lstm_model import LSTMPredictor

for run in range(2):
    model = LSTMPredictor()
    ok, msg = model.train(X[:200], y[:200], hidden_size=32, num_layers=1,
                          seq_len=10, epochs=20, batch_size=16, test_size=0.2)
    if ok:
        print(f"  Run{run+1}: R2={model.metrics['R2']:.4f}")
        results.setdefault('LSTM_small', []).append(model.metrics['R2'])
    else:
        print(f"  Run{run+1}: FAILED")

# Test 3: Small dataset - PatchTST
print("\n--- PatchTST (small data 200 samples) ---")
from src.models.patchtst_model import PatchTSTPredictor

for run in range(2):
    model = PatchTSTPredictor()
    ok, msg = model.train(X[:200], y[:200], seq_len=24, pred_len=12,
                          d_model=32, n_layers=1, n_heads=2, epochs=20, batch_size=8, test_size=0.2)
    if ok:
        print(f"  Run{run+1}: R2={model.metrics['R2']:.4f}")
        results.setdefault('PatchTST_small', []).append(model.metrics['R2'])
    else:
        print(f"  Run{run+1}: FAILED")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for name, vals in sorted(results.items()):
    avg = np.mean(vals)
    status = "OK" if avg > -1.0 else "NEEDS WORK"
    print(f"  {status} {name:20s}: avg_R2={avg:+.4f}")

print("\nTest complete!")
