"""
DeepDetect 调试测试脚本
直接测试所有检测器，不依赖 Gradio
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core.data_loader import DataLoader
from src.core.eval import evaluate_detector, format_metrics_table, find_anomaly_intervals, format_anomaly_intervals
from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.ocsvm import OCSVMDetector
from src.models.lof_detector import LOFDetector
from src.models.stats_detector import StatsDetector
from src.models.lstm_detector import LSTMDetector

# ============ 1. 加载数据集 ============
print("=" * 60)
print("【1. 加载数据集】")
print("=" * 60)

csv_path = r"C:\Users\XJH\DeepResearch\test_data\dd_temp_anomaly.csv"
dl = DataLoader()
success, msg = dl.load_csv(csv_path)
print(f"加载结果: {msg}")

df = dl.df
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print(f"数据类型:\n{df.dtypes}")
print(f"\n前5行:\n{df.head()}")
print(f"\n统计摘要:\n{df['value'].describe()}")

# 提取 value 列
values = df['value'].values.astype(np.float32)
print(f"\nvalue 列: {len(values)} 个样本")

# ============ 2. 测试 IsolationForest ============
print("\n" + "=" * 60)
print("【2. 测试 IsolationForest】")
print("=" * 60)

X_2d = values.reshape(-1, 1)
if_model = IsolationForestDetector(contamination=0.1)
if_model.fit(X_2d)
if_scores = if_model.score_samples(X_2d)
if_labels = if_model.predict(X_2d)
if_threshold = if_model.get_threshold()

n_anomalies = int(np.sum(if_labels))
print(f"阈值: {if_threshold:.4f}")
print(f"检测到异常数: {n_anomalies} / {len(if_labels)} ({n_anomalies/len(if_labels)*100:.2f}%)")
print(f"分数统计: mean={np.mean(if_scores):.4f}, std={np.std(if_scores):.4f}, max={np.max(if_scores):.4f}")

# 异常区间
if_intervals = find_anomaly_intervals(if_labels)
print(f"\n连续异常区间数: {len(if_intervals)}")
if len(if_intervals) > 0:
    print(f"区间摘要: 总点数={sum(e-s+1 for s,e in if_intervals)}, 最长={max(e-s+1 for s,e in if_intervals)}, 最短={min(e-s+1 for s,e in if_intervals)}")
    print(format_anomaly_intervals(if_intervals, if_scores))

# ============ 3. 测试 LSTM ============
print("\n" + "=" * 60)
print("【3. 测试 LSTM 检测器】")
print("=" * 60)

try:
    lstm_model = LSTMDetector(
        contamination=0.1,
        seq_len=20,
        hidden_size=64,
        num_layers=2,
        epochs=30,  # 减少epoch加速测试
        batch_size=64,
        device='cpu'
    )
    lstm_model.fit(X_2d)
    lstm_scores = lstm_model.score_samples(X_2d)
    lstm_labels = lstm_model.predict(X_2d)
    lstm_threshold = lstm_model.get_threshold()

    n_lstm_anomalies = int(np.sum(lstm_labels))
    print(f"LSTM 阈值: {lstm_threshold:.4f}")
    print(f"LSTM 检测到异常数: {n_lstm_anomalies} / {len(lstm_labels)} ({n_lstm_anomalies/len(lstm_labels)*100:.2f}%)")
    print(f"分数范围: [{np.min(lstm_scores):.4f}, {np.max(lstm_scores):.4f}]")

    lstm_intervals = find_anomaly_intervals(lstm_labels)
    print(f"\nLSTM 连续异常区间数: {len(lstm_intervals)}")
except Exception as e:
    print(f"LSTM 测试失败: {e}")
    import traceback
    traceback.print_exc()

# ============ 4. 测试全部 7 种检测器 ============
print("\n" + "=" * 60)
print("【4. 测试全部 7 种检测器】")
print("=" * 60)

detector_results = {}

# 4a. IsolationForest
print("\n--- IsolationForest ---")
detector_results['IsolationForest'] = {
    'labels': if_labels,
    'scores': if_scores,
    'threshold': if_threshold,
    'n_anomalies': n_anomalies
}
print(f"  异常数: {n_anomalies}, 比例: {n_anomalies/len(if_labels)*100:.2f}%")

# 4b. Autoencoder
print("\n--- Autoencoder ---")
try:
    ae_model = AutoencoderDetector(contamination=0.1, epochs=30, batch_size=64)
    ae_model.fit(X_2d)
    ae_scores = ae_model.score_samples(X_2d)
    ae_labels = ae_model.predict(X_2d)
    ae_threshold = ae_model.get_threshold()
    n_ae = int(np.sum(ae_labels))
    detector_results['Autoencoder'] = {
        'labels': ae_labels, 'scores': ae_scores,
        'threshold': ae_threshold, 'n_anomalies': n_ae
    }
    print(f"  异常数: {n_ae}, 比例: {n_ae/len(ae_labels)*100:.2f}%")
except Exception as e:
    print(f"  Autoencoder 失败: {e}")
    detector_results['Autoencoder'] = {'error': str(e)}

# 4c. OneClassSVM
print("\n--- OneClassSVM ---")
try:
    ocsvm_model = OCSVMDetector(contamination=0.1)
    ocsvm_model.fit(X_2d)
    ocsvm_scores = ocsvm_model.score_samples(X_2d)
    ocsvm_labels = ocsvm_model.predict(X_2d)
    ocsvm_threshold = ocsvm_model.get_threshold()
    n_ocsvm = int(np.sum(ocsvm_labels))
    detector_results['OneClassSVM'] = {
        'labels': ocsvm_labels, 'scores': ocsvm_scores,
        'threshold': ocsvm_threshold, 'n_anomalies': n_ocsvm
    }
    print(f"  异常数: {n_ocsvm}, 比例: {n_ocsvm/len(ocsvm_labels)*100:.2f}%")
except Exception as e:
    print(f"  OneClassSVM 失败: {e}")
    detector_results['OneClassSVM'] = {'error': str(e)}

# 4d. LOF
print("\n--- LOF ---")
try:
    lof_model = LOFDetector(contamination=0.1)
    lof_model.fit(X_2d)
    lof_scores = lof_model.score_samples(X_2d)
    lof_labels = lof_model.predict(X_2d)
    lof_threshold = lof_model.get_threshold()
    n_lof = int(np.sum(lof_labels))
    detector_results['LOF'] = {
        'labels': lof_labels, 'scores': lof_scores,
        'threshold': lof_threshold, 'n_anomalies': n_lof
    }
    print(f"  异常数: {n_lof}, 比例: {n_lof/len(lof_labels)*100:.2f}%")
except Exception as e:
    print(f"  LOF 失败: {e}")
    detector_results['LOF'] = {'error': str(e)}

# 4e. Stats_ZScore
print("\n--- Stats_ZScore ---")
try:
    zscore_model = StatsDetector(method='zscore', contamination=0.1)
    zscore_model.fit(X_2d)
    zscore_scores = zscore_model.score_samples(X_2d)
    zscore_labels = zscore_model.predict(X_2d)
    zscore_threshold = zscore_model.get_threshold()
    n_zscore = int(np.sum(zscore_labels))
    detector_results['Stats_ZScore'] = {
        'labels': zscore_labels, 'scores': zscore_scores,
        'threshold': zscore_threshold, 'n_anomalies': n_zscore
    }
    print(f"  异常数: {n_zscore}, 比例: {n_zscore/len(zscore_labels)*100:.2f}%")
except Exception as e:
    print(f"  Stats_ZScore 失败: {e}")
    detector_results['Stats_ZScore'] = {'error': str(e)}

# 4f. Stats_IQR
print("\n--- Stats_IQR ---")
try:
    iqr_model = StatsDetector(method='iqr', contamination=0.1)
    iqr_model.fit(X_2d)
    iqr_scores = iqr_model.score_samples(X_2d)
    iqr_labels = iqr_model.predict(X_2d)
    iqr_threshold = iqr_model.get_threshold()
    n_iqr = int(np.sum(iqr_labels))
    detector_results['Stats_IQR'] = {
        'labels': iqr_labels, 'scores': iqr_scores,
        'threshold': iqr_threshold, 'n_anomalies': n_iqr
    }
    print(f"  异常数: {n_iqr}, 比例: {n_iqr/len(iqr_labels)*100:.2f}%")
except Exception as e:
    print(f"  Stats_IQR 失败: {e}")
    detector_results['Stats_IQR'] = {'error': str(e)}

# 4g. LSTM (already tested above)
if 'LSTM' not in detector_results:
    try:
        detector_results['LSTM'] = {
            'labels': lstm_labels, 'scores': lstm_scores,
            'threshold': lstm_threshold, 'n_anomalies': n_lstm_anomalies
        }
    except:
        pass

# ============ 5. 汇总表格 ============
print("\n" + "=" * 60)
print("【5. 检测器汇总】")
print("=" * 60)
print(f"{'检测器':<20} {'异常数':>8} {'异常比例':>10} {'阈值':>10}")
print("-" * 50)
for name, res in detector_results.items():
    if 'error' in res:
        print(f"{name:<20} {'ERROR':>8}  {res['error'][:30]}")
    else:
        pct = res['n_anomalies'] / len(values) * 100
        print(f"{name:<20} {res['n_anomalies']:>8} {pct:>9.2f}% {res['threshold']:>10.4f}")

# ============ 6. 生成可视化 ============
print("\n" + "=" * 60)
print("【6. 生成可视化】")
print("=" * 60)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# 图1: 原始时序 + IsolationForest 异常
ax = axes[0, 0]
ax.plot(range(len(values)), values, 'b-', alpha=0.5, linewidth=0.5, label='Temperature')
anomaly_idx = np.where(if_labels == 1)[0]
if len(anomaly_idx) > 0:
    ax.scatter(anomaly_idx, values[anomaly_idx], c='red', s=5, alpha=0.7, label=f'Anomalies ({len(anomaly_idx)})')
ax.set_title('IsolationForest - Temperature Anomalies')
ax.set_xlabel('Index')
ax.set_ylabel('Temperature')
ax.legend()
ax.grid(True, alpha=0.3)

# 图2: 异常分数
ax = axes[0, 1]
ax.plot(range(len(if_scores)), if_scores, 'b-', alpha=0.5, linewidth=0.5)
ax.axhline(y=if_threshold, color='r', linestyle='--', label=f'Threshold ({if_threshold:.3f})')
ax.scatter(anomaly_idx, if_scores[anomaly_idx], c='red', s=5, alpha=0.7)
ax.set_title('IsolationForest - Anomaly Scores')
ax.set_xlabel('Index')
ax.set_ylabel('Score')
ax.legend()
ax.grid(True, alpha=0.3)

# 图3: LSTM
if 'LSTM' in detector_results and 'error' not in detector_results['LSTM']:
    ax = axes[1, 0]
    ax.plot(range(len(values)), values, 'b-', alpha=0.5, linewidth=0.5)
    lstm_anomaly_idx = np.where(lstm_labels == 1)[0]
    if len(lstm_anomaly_idx) > 0:
        ax.scatter(lstm_anomaly_idx, values[lstm_anomaly_idx], c='orange', s=5, alpha=0.7, label=f'Anomalies ({len(lstm_anomaly_idx)})')
    ax.set_title('LSTM - Temperature Anomalies')
    ax.set_xlabel('Index')
    ax.set_ylabel('Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(range(len(lstm_scores)), lstm_scores, 'b-', alpha=0.5, linewidth=0.5)
    ax.axhline(y=lstm_threshold, color='r', linestyle='--', label=f'Threshold ({lstm_threshold:.3f})')
    ax.scatter(lstm_anomaly_idx, lstm_scores[lstm_anomaly_idx], c='orange', s=5, alpha=0.7)
    ax.set_title('LSTM - Anomaly Scores')
    ax.set_xlabel('Index')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'LSTM not available', ha='center', va='center')
    axes[1, 1].text(0.5, 0.5, 'LSTM not available', ha='center', va='center')

# 图4: 多检测器异常点对比
ax = axes[2, 0]
ax.plot(range(len(values)), values, 'b-', alpha=0.3, linewidth=0.3, label='Temperature')
colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
detectors_2d = ['IsolationForest', 'Autoencoder', 'OneClassSVM', 'LOF', 'Stats_ZScore', 'Stats_IQR']
for i, det_name in enumerate(detectors_2d):
    if det_name in detector_results and 'error' not in detector_results[det_name]:
        anom_idx = np.where(detector_results[det_name]['labels'] == 1)[0]
        if len(anom_idx) > 0:
            offset = (i - 2.5) * 2
            ax.scatter(anom_idx, values[anom_idx] + offset, c=colors[i], s=2, alpha=0.5, label=f'{det_name} ({len(anom_idx)})')
ax.set_title('All Detectors - Anomaly Points (staggered for visibility)')
ax.set_xlabel('Index')
ax.set_ylabel('Temperature + offset')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

# 图5: 异常区间对比
ax = axes[2, 1]
det_names = []
interval_counts = []
for name, res in detector_results.items():
    if 'error' not in res:
        intervals = find_anomaly_intervals(res['labels'])
        det_names.append(name)
        interval_counts.append(len(intervals))
ax.bar(det_names, interval_counts, color=['steelblue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray'][:len(det_names)])
ax.set_title('Anomaly Interval Count per Detector')
ax.set_xlabel('Detector')
ax.set_ylabel('# Intervals')
ax.tick_params(axis='x', rotation=30)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), 'test_visualization.png')
plt.savefig(output_path, dpi=100, bbox_inches='tight')
print(f"可视化已保存: {output_path}")
plt.close()

# ============ 7. 检查 run_detect.py 启动 ============
print("\n" + "=" * 60)
print("【7. 检查 run_detect.py】")
print("=" * 60)
with open(r"C:\Users\XJH\DeepResearch\DeepDetect\run_detect.py", encoding="utf-8") as f:
    content = f.read()
print("run_detect.py 内容:")
print(content[:500])

print("\n" + "=" * 60)
print("【调试完成】")
print("=" * 60)
