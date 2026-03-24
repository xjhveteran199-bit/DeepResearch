"""
CNN1D 聚焦优化脚本 v2 - 直接传 1D 时序，让 train() 内部构建序列
"""
import sys, pandas as pd, numpy as np, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, r'C:\Users\XJH\DeepPredict\src')
from models.cnn1d_model import CNN1DPredictorV4

df = pd.read_csv(r'C:\Users\XJH\DeepResearch\test_data\dp_temperature.csv')
temp = df['Temp'].values.astype(np.float32)

print('=== CNN1D 聚焦优化 v2 ===')
print('Data: %d days daily min temperature' % len(temp))

# 直接传 1D 时序，让 train() 内部处理 seq_len
# train() 的 train() 会对 1D 输入 reshape 成 (n, 1) 然后构建 (n_seqs, seq_len, 1)
configs = [
    # (seq_len, hidden_channels, num_layers, kernel_size, epochs, lr)
    (30, 64, 2, 3, 100, 0.001),
    (60, 64, 2, 3, 100, 0.001),
    (90, 64, 3, 3, 100, 0.001),
    (180, 128, 3, 5, 100, 0.001),
    (180, 128, 3, 5, 150, 0.001),
    (365, 128, 3, 7, 150, 0.001),
    (365, 128, 3, 7, 200, 0.0005),
    (365, 256, 3, 7, 150, 0.001),
    (365, 256, 3, 7, 200, 0.0005),
]

print('Building sequences internally...')

best_r2 = -999
best_config = {}

for seq_len, hidden_channels, num_layers, kernel_size, epochs, lr in configs:
    print('seq=%d hidden=%d layers=%d kernel=%d epochs=%d lr=%.4f ... ' % (
        seq_len, hidden_channels, num_layers, kernel_size, epochs, lr), end='', flush=True)

    try:
        model = CNN1DPredictorV4()
        ok, msg = model.train(
            temp, temp,  # X=1D时序, y=1D时序 (内部处理滑动窗口)
            seq_len=seq_len,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            epochs=epochs,
            batch_size=32,
            learning_rate=lr,
            test_size=0.1,
            pred_len=1,
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
                'epochs': epochs, 'learning_rate': lr,
                'r2': r2, 'rmse': rmse
            }
            print('  *** NEW BEST ***')

    except Exception as e:
        print('ERROR: %s' % str(e)[:100])
        import traceback
        traceback.print_exc()

print()
print('=== Best Configuration ===')
for k, v in best_config.items():
    print('%s: %s' % (k, v))
print('Best R2: %.4f' % best_r2)
print('Status: PASS (R2>0.55)' if best_r2 > 0.55 else 'Still below 0.55 target')
