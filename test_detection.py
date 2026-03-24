import sys
sys.path.insert(0, 'C:/Users/XJH/DeepResearch/DeepDetect')
import pandas as pd
import numpy as np
from src.models.autoencoder import AutoencoderDetector
from src.visualizer import DetectVisualizer

# 加载数据
df = pd.read_csv('C:/Users/XJH/DeepResearch/test_data/dd_temp_anomaly.csv')
values = df['value'].values if 'value' in df.columns else df.iloc[:, 0].values
print(f'Data shape: {len(values)} rows')

# 转换为 numpy 数组
values = np.array(values, dtype=np.float32)

# 转换为 2D 数组 (n_samples, n_features)
X = values.reshape(-1, 1).astype(np.float32)

# 检测
detector = AutoencoderDetector(contamination=0.1, epochs=30)
detector.fit(X)
scores = detector.score_samples(X)
threshold = detector.get_threshold()
labels = detector.predict(X)
print(f'Scores computed: min={scores.min():.4f}, max={scores.max():.4f}, threshold={threshold:.4f}')
print(f'Anomalies detected: {labels.sum()} / {len(labels)}')

# 生成图表 - 使用 numpy.arange 而不是 range
times = np.arange(len(values))
viz = DetectVisualizer()
viz.plot_anomaly_timeseries(
    times, values,
    labels=labels,
    scores=scores, threshold=threshold,
    save_path='C:/Users/XJH/DeepResearch/dd_test_anomaly.png'
)
viz.plot_score_distribution(scores, threshold, save_path='C:/Users/XJH/DeepResearch/dd_test_scores.png')
print('✅ 异常检测图表生成成功')
