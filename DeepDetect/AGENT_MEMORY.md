# DeepDetect Agent 专属记忆

## Agent 身份
- **职责**：全职负责 DeepDetect 模块的迭代更新
- **上级**：主 Agent（汇总 + 监控）
- **报告频率**：每 2 小时向主 Agent 汇报一次

## 模块基本信息
- **模块路径**：`C:\Users\XJH\DeepResearch\DeepDetect\`
- **版本**：v1.0（2026-03-24）
- **定位**：时序异常检测，支持有监督和无监督两种模式
- **Git 远程**：`https://github.com/xjhveteran199-bit/DeepResearch`

## 当前版本功能
- 6种检测器：IsolationForest / Autoencoder / OneClassSVM / LOF / Z-score / IQR
- Autoencoder：PyTorch（Encoder: input→16→8, Decoder: 8→16→input）
- 支持有标签（计算 Precision/Recall/F1）和无标签（纯异常分数）两种模式
- 时序图标注异常点、异常分数可视化、结果 CSV 导出
- Web 界面：Gradio（端口7861独立运行）

## 技术栈
- Python 3.12 + PyTorch 2.2.0 + scikit-learn + Gradio 6.9.0
- numpy 1.26.4（注意：torch 2.2.0 需要 numpy < 2.0）

## 已知问题
- Autoencoder 对高维数据训练较慢，建议 epoch 不超过 200
- IsolationForest 的 contamination 参数需要预先估计异常比例

## 迭代 Roadmap
1. 增加 LSTM-based 异常检测器（时序专用）
2. 增加多变量异常检测（多列同时输入）
3. 增加滑动窗口异常检测（局部异常 vs 全局异常）
4. 增加异常区间检测（连续异常段识别）
5. 增加 SHAP 可解释性（哪些特征导致异常）
6. API 服务化（FastAPI）

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
| 2026-03-24 | v1.0 初始版本，6种检测器完整实现 | ✅ 完成 |
