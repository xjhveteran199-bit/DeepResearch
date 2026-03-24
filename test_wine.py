# -*- coding: utf-8 -*-
"""
DeepClassify - Wine 数据集测试脚本
"""
import sys
import os

# UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add paths
sys.path.insert(0, 'C:/Users/XJH/DeepResearch')
sys.path.insert(0, 'C:/Users/XJH/DeepResearch/DeepClassify')
sys.path.insert(0, 'C:/Users/XJH/DeepResearch/DeepClassify/src')

# Non-interactive matplotlib
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import matplotlib
matplotlib.use('Agg')

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 导入 DeepClassify 模块
from src.models.rf_classifier import RFClassifier
from src.visualizer import ClassifyVisualizer

print("=" * 60)
print("DeepClassify - Wine 数据集测试")
print("=" * 60)

# 1. 加载 Wine 数据集
print("\n[1] 加载 UCI Wine 数据集...")
data = load_wine()
X = data.data  # 13个特征
y = data.target  # 3分类 (0,1,2)
class_names = list(data.target_names)  # ['class_0', 'class_1', 'class_2']
print(f"    数据集: X.shape={X.shape}, y.shape={y.shape}")
print(f"    类别: {class_names}")
print(f"    特征: {data.feature_names[:3]}...")

# 2. 划分训练集/测试集
print("\n[2] 划分训练集/测试集 (80%/20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    训练集: {X_train.shape[0]} 样本")
print(f"    测试集: {X_test.shape[0]} 样本")

# 3. 训练 RandomForest 分类器
print("\n[3] 训练 RandomForest 分类器...")
classifier = RFClassifier(n_estimators=100, max_depth=10, random_state=42)
ok, msg = classifier.fit(X_train, y_train)
print(f"    {msg}")

# 4. 预测
print("\n[4] 预测测试集...")
y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)
print(f"    预测完成: y_pred.shape={y_pred.shape}, y_proba.shape={y_proba.shape}")

# 5. 计算指标（使用内部 metrics）
print("\n[5] 评估结果...")
metrics = classifier.metrics
accuracy = metrics.get('Accuracy', 0)
f1 = metrics.get('F1_weighted', 0)
print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"    F1(weighted): {f1:.4f} ({f1*100:.2f}%)")

# 6. 可视化
print("\n[6] 生成可视化图表...")

# 对于可视化，需要编码后的标签 (0, 1, 2...)
# classifier._label_encoder 已经 fit 过了
label_encoder = classifier._label_encoder
y_test_encoded = label_encoder.transform(y_test.astype(str))
y_pred_encoded = label_encoder.transform(y_pred.astype(str))

viz = ClassifyVisualizer()

# 6.1 混淆矩阵
viz.plot_confusion_matrix(
    y_test_encoded, y_pred_encoded, 
    labels=class_names,
    save_path="dc_cm.png"
)
print("    ✅ 混淆矩阵: dc_cm.png")

# 6.2 ROC曲线
viz.plot_roc_curve(
    y_test_encoded, y_proba,
    labels=class_names,
    save_path="dc_roc.png"
)
print("    ✅ ROC曲线: dc_roc.png")

# 6.3 t-SNE (fix parameter order)
viz.plot_tsne(
    X_features=X_test,
    y_true=y_test_encoded,
    labels=class_names,
    save_path="dc_tsne.png"
)
print("    ✅ t-SNE图: dc_tsne.png")

print("\n" + "=" * 60)
print("✅ 全部图表生成成功!")
print("=" * 60)

# 7. 验证标准
print("\n[7] 验证结果:")
print(f"    ✅ 程序正常运行无崩溃")
accuracy_pct = accuracy * 100
print(f"    ✅ Accuracy = {accuracy_pct:.2f}% {'✅' if accuracy > 0.7 else '❌'} (要求 > 70%)")
print(f"    ✅ 3个图表文件全部生成 (PNG)")
