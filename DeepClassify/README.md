# DeepClassify - AI 信号分类模块

> Deep-Research 平台的信号分类模块，支持 CNN1D、RandomForest、GradientBoosting、SVM 四种分类器。

## 模块功能

- **数据加载**：CSV 上传，自动识别数值/类别/日期列，缺失值填充，标准化
- **特征工程**：自动列类型检测，日期特征提取（年/月/日/星期/周期编码）
- **模型训练**：CNN1D / RandomForest / GradientBoosting / SVM
- **评估指标**：Accuracy / Precision / Recall / F1-score（宏观/微观/加权）、混淆矩阵、ROC-AUC / PR-AUC
- **模型保存**：导出 `.pkl` 模型文件，支持加载复用
- **批量预测**：对新数据进行分类预测

## 目录结构

```
DeepClassify/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data_loader.py    # CSV 数据加载与预处理
│   │   └── metrics.py        # 分类评估指标（Accuracy/F1/ROC/Confusion Matrix）
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier_base.py    # 分类器基类
│   │   ├── cnn1d_classifier.py   # CNN1D 一维卷积神经网络分类器
│   │   ├── rf_classifier.py      # RandomForest 随机森林分类器
│   │   ├── gb_classifier.py      # GradientBoosting 梯度提升分类器
│   │   └── svm_classifier.py     # SVM 支持向量机分类器
│   └── utils/
│       ├── __init__.py
├── app.py               # Gradio Web 界面
├── run_classify.py      # 入口脚本
├── requirements.txt     # 依赖列表
└── README.md
```

## 支持的模型

| 模型 | 类型 | 适用场景 |
|------|------|----------|
| **CNN1D** | 深度学习 | 信号/时序分类（心电、肌电、手势识别等） |
| **RandomForest** | 传统 ML | 通用分类，特征重要性分析 |
| **GradientBoosting** | 传统 ML | 高精度分类（XGBoost/LightGBM/sklearn 后端自动选择） |
| **SVM** | 传统 ML | 核方法，小样本高维数据 |

## CNN1D 架构说明

- 输入：`reshape(samples, 1, n_features)` — 单通道一维信号
- 若特征数 < 8，自动 padding 到 8
- 结构：`Conv1D → BatchNorm → ReLU → MaxPool → Conv1D → BatchNorm → ReLU → MaxPool → GlobalAvgPool → FC → classes`
- 支持 Early Stopping

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 启动 Web 界面

```bash
python run_classify.py
```

浏览器打开 `http://localhost:7861`

### Web 操作流程

1. **数据加载**：上传含特征列和标签列的 CSV 文件
2. **特征与标签**：选择目标列（标签），调整测试集比例
3. **选择模型**：选择分类模型（CNN1D / RF / GBM / SVM），设置参数
4. **训练与评估**：点击训练，查看 Accuracy / F1 / Confusion Matrix / ROC 曲线
5. **预测新数据**：上传待预测 CSV，输出分类结果

### 代码调用示例

```python
from src.core.data_loader import DataLoader
from src.models import CNN1DClassifyWrapper, RFClassifier

# 加载数据
loader = DataLoader()
loader.load_csv("data.csv")
loader.select_target("label")
X = loader.get_feature_matrix(exclude_cols=["label"])
y, le = loader.get_target_encoded()

# 训练 CNN1D
clf = CNN1DClassifyWrapper(epochs=50, hidden_channels=64)
ok, msg = clf.fit(X, y)
print(msg)

# 预测
pred = clf.predict(X[:10])
print("预测结果:", pred)

# 保存模型
clf.save("model.pkl")
```

## 评估指标说明

- **Accuracy**：整体正确率
- **Precision (weighted)**：加权精确率
- **Recall (weighted)**：加权召回率
- **F1-score (weighted)**：加权 F1 分数
- **ROC-AUC**：二分类/多分类 ROC 曲线下面积
- **PR-AUC**：精确率-召回率曲线下面积
- **Confusion Matrix**：混淆矩阵，行=真实类别，列=预测类别

## 与 DeepPredict 的关系

DeepClassify 是 Deep-Research 平台的**分类模块**，与预测模块 DeepPredict 互补：

- **DeepPredict**：回归/时序预测（如股价、温度预测）
- **DeepClassify**：分类/信号识别（如心电分类、手势识别、故障检测）
