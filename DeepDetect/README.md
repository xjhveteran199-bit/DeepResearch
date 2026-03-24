# DeepDetect - 异常检测模块

> Deep-Research 时序/分类/检测 AI 分析平台的一部分

## 模块功能

DeepDetect 是一个基于 Python 的异常检测系统，支持：

- **无监督检测**：无标签数据，使用全部数据训练
- **有监督评估**：有标签数据，计算 Precision/Recall/F1 等指标
- **多种检测方法**：IsolationForest、Autoencoder、OneClassSVM、LOF、Z-score、IQR
- **Web 可视化**：Gradio 交互界面，时序图标注、分数分布、箱线图
- **结果导出**：CSV 导出带异常标签和分数

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Web 界面

```bash
python run_detect.py
```

访问 http://localhost:7861

### 3. 使用流程

1. **数据加载 Tab**：上传 CSV 文件（支持 utf-8/gbk/gb2312 编码）
2. **检测配置 Tab**：选择检测方法、设置参数、点击"开始检测"
3. **检测结果 Tab**：查看可视化图表
4. **导出结果 Tab**：下载带异常标签的 CSV

## 支持的检测方法

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| **IsolationForest** | 随机森林隔离方法 | 通用大规模数据 |
| **Autoencoder** | PyTorch 自编码器（重建误差） | 复杂非线性数据 |
| **OneClassSVM** | 单类支持向量机 | 中小规模数据 |
| **LOF** | 局部异常因子（密度） | 多维数据 |
| **Stats_ZScore** | Z-score 统计方法 | 简单单维数据 |
| **Stats_IQR** | 四分位距方法 | 简单单维数据 |

## Autoencoder 架构

```
Encoder: Linear(输入维度) → ReLU → Linear(16) → ReLU → Linear(8)
Decoder: Linear(8) → ReLU → Linear(16) → ReLU → Linear(输入维度)
异常判定：reconstruction_error > threshold（如95%分位数）
```

## 评估指标

- **有标签模式**：Precision / Recall / F1 / AUC-ROC / AUC-PR / 检测率 / 误报率
- **无标签模式**：分数统计（均值/标准差/分位数）

## 目录结构

```
DeepDetect/
├── app.py                    # Gradio Web 界面
├── run_detect.py             # 入口脚本
├── requirements.txt          # Python 依赖
├── README.md                 # 本文件
└── src/
    ├── core/
    │   ├── __init__.py
    │   ├── data_loader.py     # 数据加载模块
    │   └── eval.py            # 评估指标模块
    └── models/
        ├── __init__.py
        ├── detector_base.py   # 检测器基类
        ├── isolation_forest.py
        ├── autoencoder.py     # PyTorch 自编码器
        ├── ocsvm.py           # One-Class SVM
        ├── lof_detector.py    # LOF
        └── stats_detector.py  # Z-score / IQR
```

## 技术栈

- Python 3.12+
- PyTorch (Autoencoder)
- scikit-learn (IsolationForest / OCSVM / LOF)
- Gradio (Web 界面)
- Pandas / NumPy / Matplotlib
