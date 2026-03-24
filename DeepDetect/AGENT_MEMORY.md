# DeepDetect Agent 专属记忆

## Agent 身份
- **职责**：全职负责 DeepDetect 模块的迭代更新
- **上级**：主 Agent（汇总 + 监控）
- **报告频率**：每 2 小时向主 Agent 汇报一次

## 模块基本信息
- **模块路径**：`C:\Users\XJH\DeepResearch\DeepDetect\`
- **版本**：v1.1（2026-03-24）
- **定位**：时序异常检测，支持有监督和无监督两种模式
- **Git 远程**：`https://github.com/xjhveteran199-bit/DeepResearch`

## 当前版本功能（v1.1）
- **7种检测器**：IsolationForest / Autoencoder / OneClassSVM / LOF / LSTM / Z-score / IQR
- **Autoencoder**：PyTorch（Encoder: input→16→8, Decoder: 8→16→input）
- **LSTM 异常检测器**（新增 v1.1）：基于 PyTorch LSTM 预测误差的时序异常检测，支持单变量和多变量
  - 参数：seq_len（窗口长度）、hidden_size、num_layers、dropout、epochs
  - 适用场景：时序数据、传感器数据、金融数据
- 支持有标签（计算 Precision/Recall/F1）和无标签（纯异常分数）两种模式
- 时序图标注异常点、异常分数可视化、结果 CSV 导出
- **滑动窗口异常区间检测**（新增 v1.1）：`find_anomaly_intervals()` 提取连续异常段
- Web 界面：Gradio（端口7861独立运行）

## 技术栈
- Python 3.12 + PyTorch 2.2.0（CPU）+ scikit-learn + Gradio 6.9.0
- numpy 1.26.4（注意：torch 2.2.0 需要 numpy < 2.0）

## 已知问题
- Autoencoder 对高维数据训练较慢，建议 epoch 不超过 200
- IsolationForest 的 contamination 参数需要预先估计异常比例
- PyTorch CPU 版本在某些 Windows 环境下有 DLL 加载问题（1114/126 错误）
  - 解决：清理 `~orch` 等残留目录后重新安装 torch

## 迭代 Roadmap
1. ✅ 增加 LSTM-based 异常检测器（时序专用）- v1.1 完成
2. ✅ 增加滑动窗口异常区间检测 - v1.1 完成
3. 增加多变量异常检测（多列同时输入）- 已部分支持（LSTM/Autoencoder）
4. 增加 SHAP 可解释性（哪些特征导致异常）
5. API 服务化（FastAPI）

## 新增文件
- `src/models/lstm_detector.py`：LSTM 时序异常检测器
- `src/core/eval.py`：新增 `find_anomaly_intervals`、`format_anomaly_intervals`、`evaluate_with_intervals`

## 修改文件
- `app.py`：注册 LSTM 检测器，新增 seq_len/hidden_size UI 参数
- `src/core/eval.py`：新增滑动窗口异常区间函数

## 汇报模板
每次汇报请包含：
```
## [时间] DeepDetect 进展汇报

### 本次更新
- 做了什么
- 修改了哪些文件

### 代码状态
- 语法检查：✅/❌
- 功能测试：✅/❌

### 下一步计划

### 问题/需要协助
```

## 最近更新记录
| 日期 | 更新内容 | 状态 |
|------|---------|------|
| 2026-03-24 | 独立调试测试（dd_temp_anomaly.csv，7267行）：全部7种检测器通过 | ✅ 完成 |
| 2026-03-24 | BUG修复：LOFDetector 的 `decision_function` 缺失 → 添加 `novelty=True` | ✅ 完成 |
| 2026-03-24 | v1.1 新增 LSTM 时序异常检测器 + 滑动窗口异常区间检测 | ✅ 完成 |
| 2026-03-24 | v1.0 初始版本，6种检测器完整实现 | ✅ 完成 |
