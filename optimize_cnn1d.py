"""
CNN1D 参数优化脚本
利用 CNN1DPredictorV4.train() 的内部评估（避免 predict horizon 问题）
"""
import sys, pandas as pd, numpy as np, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, r'C:\Users\XJH\DeepPredict\src')
from models.cnn1d_model import CNN1DPredictorV4

df = pd.read_csv(r'C:\Users\XJH\DeepResearch\test_data\dp_temperature.csv')
temp = df['Temp'].values.astype(np.float32)

print('=== CNN1D 参数优化 ===')
print('Data: 3650 days daily min temperature')

best_r2 = -999
best_config = {}

configs = [
    # (seq_len, hidden_channels, num_layers, kernel_size, epochs, pred_len)
    (30, 64, 2, 3, 50, 1),
    (60, 64, 2, 3, 50, 1),
    (90, 64, 3, 3, 50, 1),
    (90, 128, 3, 3, 50, 1),
    (90, 64, 3, 5, 80, 1),
    (180, 64, 3, 3, 80, 1),
    (180, 128, 3, 5, 80, 1),
    (365, 64, 3, 7, 100, 1),  # full year context
]

for seq_len, hidden_channels, num_layers, kernel_size, epochs, pred_len in configs:
    print('seq=%d hidden=%d layers=%d kernel=%d epochs=%d ... ' % (
        seq_len, hidden_channels, num_layers, kernel_size, epochs), end='', flush=True)

    try:
        # Build sequences
        X_seq, y_seq = [], []
        for i in range(seq_len, len(temp)):
            X_seq.append(temp[i-seq_len:i].flatten())
            y_seq.append(temp[i])
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)

        n_train = int(len(X_seq) * 0.9)
        X_tr, X_te = X_seq[:n_train], X_seq[n_train:]
        y_tr, y_te = y_seq[:n_train], y_seq[n_train:]

        model = CNN1DPredictorV4()
        ok, msg = model.train(
            X_tr, y_tr,
            seq_len=seq_len,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            epochs=epochs,
            batch_size=32,
            learning_rate=0.001,
            test_size=0.1,
            pred_len=pred_len,
            verbose=False
        )

        r2 = model.metrics.get('R2', -999)
        rmse = model.metrics.get('RMSE', -999)
        print('R2=%.4f RMSE=%.4f' % (r2, rmse))

        if r2 > best_r2:
            best_r2 = r2
            best_config = {
                'seq_len': seq_len, 'hidden_channels': hidden_channels,
                'num_layers': num_layers, 'kernel_size': kernel_size,
                'epochs': epochs, 'pred_len': pred_len,
                'r2': r2, 'rmse': rmse
            }
            print('  *** NEW BEST ***')

    except Exception as e:
        print('ERROR: %s' % str(e)[:100])

print()
print('=== Best Configuration ===')
for k, v in best_config.items():
    print('%s: %s' % (k, v))
print('Best R2: %.4f' % best_r2)
print('Status: PASS (R2>0.55)' if best_r2 > 0.55 else 'Still below 0.55 target')
