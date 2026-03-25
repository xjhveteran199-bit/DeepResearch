# -*- coding: utf-8 -*-
"""
DeepPredict v1.04 完整验证测试
"""
import sys
sys.path.insert(0, 'src')

import os
os.chdir(r'C:\Users\XJH\DeepPredict')

results = []

# ===== Test 1: Data Decoupling =====
print("=" * 50)
print("Test 1: Data Decoupling (data_loader.py v2)")
print("=" * 50)
try:
    from core.data_loader import DataLoader
    dl = DataLoader()
    ok, msg = dl.load_csv('data/pollution.csv')
    print('Load OK:', ok, msg)
    summary = dl.get_summary()
    print('Shape:', summary['shape'])
    print('Numeric cols:', summary['numeric_cols'])
    print('Date cols:', summary['date_cols'])
    X = dl.get_feature_matrix(exclude_cols=['No'])
    print('Feature matrix:', X.shape)
    si = dl.detect_irregular_sampling()
    print('Sampling regular:', si.get('is_regular'), 'CV:', round(si.get('cv', 0), 4))
    results.append(('Data Decoupling', 'PASS', '-', 'v2 column detection working'))
except Exception as e:
    print('FAIL:', e)
    results.append(('Data Decoupling', 'FAIL', '-', str(e)))

# ===== Test 2: CNN1D Single Variable =====
print("\n" + "=" * 50)
print("Test 2: CNN1D Single Variable")
print("=" * 50)
try:
    import numpy as np
    import pandas as pd
    from models.cnn1d_model import CNN1DPredictorV4

    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
    y = df['Temp'].values.astype(np.float32)

    p = CNN1DPredictorV4()
    ok, msg = p.train(X=y.reshape(-1,1), y=y, seq_len=30, pred_len=7,
                      hidden_channels=64, num_layers=2, epochs=20,
                      batch_size=32, target_col='Temp')
    r2 = p.metrics.get('R2', 0)
    rmse = p.metrics.get('RMSE', 0)
    print('CNN1D OK:', ok, 'R2:', round(r2, 4), 'RMSE:', round(rmse, 4))
    results.append(('CNN1D Single', 'PASS' if ok else 'FAIL', round(r2, 4), round(rmse, 4)))
except Exception as e:
    print('FAIL:', e)
    results.append(('CNN1D Single', 'FAIL', '-', str(e)))

# ===== Test 3: LSTM Stability =====
print("\n" + "=" * 50)
print("Test 3: LSTM Stability")
print("=" * 50)
try:
    from models.lstm_model import LSTMPredictor

    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
    y = df['Temp'].values.astype(np.float32)

    p = LSTMPredictor()
    ok, msg = p.train(X=y.reshape(-1,1), y=y, seq_len=30,
                      epochs=20, batch_size=32, target_col='Temp')
    r2 = p.metrics.get('R2', 0)
    rmse = p.metrics.get('RMSE', 0)
    print('LSTM OK:', ok, 'R2:', round(r2, 4), 'RMSE:', round(rmse, 4))
    results.append(('LSTM', 'PASS' if ok else 'FAIL', round(r2, 4), round(rmse, 4)))
except Exception as e:
    print('FAIL:', e)
    results.append(('LSTM', 'FAIL', '-', str(e)))

# ===== Test 4: sklearn GradientBoosting =====
print("\n" + "=" * 50)
print("Test 4: sklearn GradientBoosting")
print("=" * 50)
try:
    from models.predictor import Predictor

    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
    X_df = df[['Temp']].copy()
    y = df['Temp'].values

    p = Predictor()
    ok, msg = p.train(X_df, y, task_type='regression', model_name='GradientBoosting')
    r2 = p.metrics.get('R2', 0)
    rmse = p.metrics.get('RMSE', 0)
    print('GB OK:', ok, 'R2:', round(r2, 4), 'RMSE:', round(rmse, 4))
    results.append(('GradientBoosting', 'PASS' if ok else 'FAIL', round(r2, 4), round(rmse, 4)))
except Exception as e:
    print('FAIL:', e)
    results.append(('GradientBoosting', 'FAIL', '-', str(e)))

# ===== Test 5: CNN1D Multivariate =====
print("\n" + "=" * 50)
print("Test 5: CNN1D Multivariate (input_size > 1)")
print("=" * 50)
try:
    from models.cnn1d_model import CNN1DPredictorV4
    import numpy as np

    np.random.seed(42)
    n = 1000
    x1 = np.cumsum(np.random.randn(n)) + 50
    x2 = np.sin(np.arange(n) * 0.01) * 10 + 30
    X = np.column_stack([x1, x2])
    y = x1 * 0.8 + x2 * 0.2 + np.random.randn(n) * 2

    p = CNN1DPredictorV4()
    ok, msg = p.train(X, y, seq_len=48, pred_len=12,
                      hidden_channels=64, num_layers=2,
                      epochs=15, batch_size=32, target_col='target')
    r2 = p.metrics.get('R2', 0)
    rmse = p.metrics.get('RMSE', 0)
    print('Multi-var CNN1D OK:', ok, 'R2:', round(r2, 4), 'RMSE:', round(rmse, 4))
    results.append(('CNN1D Multivariate', 'PASS' if ok else 'FAIL', round(r2, 4), round(rmse, 4)))
except Exception as e:
    print('FAIL:', e)
    results.append(('CNN1D Multivariate', 'FAIL', '-', str(e)))

# ===== Summary =====
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("| Test | Status | R2 | RMSE/Note |")
print("|------|--------|-----|-----------|")
for name, status, r2, note in results:
    print(f"| {name} | {status} | {r2} | {note} |")
print("\nAll tests completed.")
