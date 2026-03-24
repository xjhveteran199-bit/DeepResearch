# Deep-Research 统一平台

**定位**：面向研究者的低门槛深度学习工具箱
**版本**：v2.0
**更新**：2026-03-24

---

## 三大模块（全部完成 ✅）

### 📈 DeepPredict — 时序 / 回归预测
- **路径**：`C:\Users\XJH\DeepPredict\`
- **支持模型**：CNN1D、PatchTST、LSTM、RandomForest、GradientBoosting、LinearRegression
- **v2.0 新增功能**：
  - ✅ **多模态输入**：支持选择多个 X 特征列 → 预测 Y 目标列
  - ✅ **SHAP 可解释性**：特征重要性分析、beeswarm 图（参考 adi6492 论文）
  - ✅ 一键下载结果包（CSV + 指标 + SHAP 图）

### 🏷️ DeepClassify — 信号 / 数据分类
- **路径**：`C:\Users\XJH\DeepResearch\DeepClassify\`
- **支持模型**：CNN1D（PyTorch）、RandomForest、GradientBoosting、SVM
- **评估指标**：Accuracy / F1 / Confusion Matrix / ROC-AUC / PR-AUC
- **CNN1D 架构**：Conv1D×2 + BatchNorm + ReLU + MaxPool + GlobalAvgPool + FC

### 🔍 DeepDetect — 异常检测
- **路径**：`C:\Users\XJH\DeepResearch\DeepDetect\`
- **支持方法**：IsolationForest、Autoencoder（PyTorch）、One-Class SVM、LOF、Z-score、IQR
- **功能**：时序图标注异常点、异常分数可视化、结果 CSV 导出

---

## 快速启动

### 统一平台（推荐）✅ 三个模块一个页面
```bash
python C:\Users\XJH\DeepResearch\deep_research_web.py
# 访问 http://localhost:7860
```

### 各模块独立运行
```bash
# DeepPredict（独立）
cd C:\Users\XJH\DeepPredict
python run_web.py

# DeepClassify（独立）
cd C:\Users\XJH\DeepResearch\DeepClassify
python run_classify.py

# DeepDetect（独立）
cd C:\Users\XJH\DeepResearch\DeepDetect
python run_detect.py
```

---

## 目录结构

```
DeepResearch/
├── deep_research_web.py     # 统一平台 Web 界面（v2.0，端口 7860）
├── README.md
│
├── DeepPredict/             # 时序预测模块
│   ├── src/
│   │   ├── models/         # CNN1D / LSTM / PatchTST
│   │   └── utils/
│   │       └── shap_analyzer.py  # SHAP 分析（v2.0 新增）
│   ├── deeppredict_web.py   # 原始独立 Web
│   └── requirements.txt
│
├── DeepClassify/            # 信号分类模块
│   ├── app.py              # Gradio 独立界面
│   ├── run_classify.py     # 入口脚本
│   ├── requirements.txt
│   └── src/
│       ├── core/           # data_loader / metrics
│       └── models/         # CNN1D / RF / GB / SVM
│
└── DeepDetect/             # 异常检测模块
    ├── app.py              # Gradio 独立界面
    ├── run_detect.py       # 入口脚本
    ├── requirements.txt
    └── src/
        ├── core/           # data_loader / eval
        └── models/         # IsolationForest / Autoencoder / OCSVM / LOF / stats
```

---

## 升级记录

### v2.0（2026-03-24）
- **项目更名**：DeepPredict → Deep-Research（统一平台）
- **多模态输入**：DeepPredict 支持多 X 特征列选择
- **SHAP 可解释性**：新增 shap_analyzer.py（参考 adi6492 论文）
- **新增 DeepClassify**：CNN1D/RF/GB/SVM 信号分类模块
- **新增 DeepDetect**：6种异常检测方法
- **统一 Web**：三模块 Tab 切换（端口 7860）

---

## 技术栈

- Python 3.12
- PyTorch（CNN1D / LSTM / PatchTST / Autoencoder）
- scikit-learn（RF / GBM / SVM / IsolationForest / LOF / OCSVM）
- Gradio（Web 界面）
- SHAP（特征重要性）
- pandas / numpy / matplotlib
