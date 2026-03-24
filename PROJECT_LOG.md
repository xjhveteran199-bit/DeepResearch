# Deep-Research 项目大事记

---

## 2026-03-24（项目创建日）

### 项目初始化
- **版本**：v1.01 → v1.1
- **事件**：DeepPredict 项目升级为 Deep-Research 统一平台，新增 DeepClassify 和 DeepDetect 两个子模块
- **GitHub**：新建仓库 https://github.com/xjhveteran199-bit/DeepResearch

### 模块构建
| 模块 | 构建方式 | 状态 |
|------|---------|------|
| DeepPredict | 主Agent构建 | ✅ 完成 |
| DeepClassify | 子Agent构建 | ✅ 完成 |
| DeepDetect | 子Agent构建 | ✅ 完成 |

### 统一网站
- 入口：`deep_research_web.py`，端口 7862
- 技术问题：Gradio 6.x State subscript bug → 已修复（改用全局变量）

### 协作架构
- **主Agent**：DeepPredict 维护 + 汇总汇报
- **子Agent-1**（DeepClassify-Agent）：8c9d6116-48b9-485d-98c4-c517f8ae895a
- **子Agent-2**（DeepDetect-Agent）：b5f67e99-9f50-422a-a900-35a792a4debd
- **监控 Cron**：每2小时汇报（id: 0f618f92）

---

## 2026-03-24（v1.1 调试日）

### 子Agent 迭代结果

#### DeepClassify v1.1
- **新增功能**：K-Fold CV、SHAP 可解释性、GB 后端选择、模型对比
- **BUG修复**：4个分类器的相对导入错误（相对import → 绝对import）
- **测试结果**：Iris 数据集，RandomForest/GB Accuracy=93.33%，K-Fold CV=94.67%
- **Commit**：`1251a5e`

#### DeepDetect v1.1
- **新增功能**：LSTM 异常检测器、滑动窗口异常区间检测
- **BUG修复**：LOFDetector novelty=False → novelty=True（修复 decision_function 崩溃）
- **测试结果**：Numenta NAB 温度异常数据，7种检测器全部通过
- **Commit**：`cbf013d`

#### DeepPredict v1.1
- **测试结果**（每日最低温度数据集）：
  - RandomForest: R²=0.610, RMSE=2.55 ✅
  - GradientBoosting: R²=0.611, RMSE=2.55 ✅
  - LSTM: R²=0.545, RMSE=2.61 ✅
  - CNN1D: R²=0.449, RMSE=2.87 ⚠️（需更多epoch）
- **Commit**：`21a3388`（Gradio State bug 修复）

---

## 2026-03-24（CNN1D 优化完成）

### CNN1D 参数优化完成
- **结果**：R²=**0.6538** ✅（远超 0.55 目标）
- **最优配置**：seq_len=180, hidden_channels=128, kernel_size=5, num_layers=3, epochs=100
- **Commit**：`a5a822b`

## 2026-03-24（开源数据验证完成）

### 三模块在线验证（开源数据 + 图表自动生成）

| 模块 | 测试数据 | 结果 | 图表 | 状态 |
|------|---------|------|------|------|
| **DeepPredict** | Airline Passengers（144行） | R²=0.9648 ✅ | dp_test_output.png (121KB) | ✅ |
| **DeepClassify** | UCI Wine（3分类，13特征） | Accuracy=100% ✅ | dc_cm.png (97KB) + dc_roc.png (124KB) + dc_tsne.png (98KB) | ✅ |
| **DeepDetect** | 温度异常数据（7267行） | 检测出异常 ✅ | dd_test_anomaly.png (754KB) + dd_test_scores.png (118KB) | ✅ |

### 图表自动生成验证
- ✅ PredictVisualizer → 预测时序图自动生成
- ✅ ClassifyVisualizer → 混淆矩阵 + ROC + t-SNE 自动生成
- ✅ DetectVisualizer → 异常时序标注 + 分数分布自动生成

### Git 提交
- `4405805` test: add Wine dataset test with visualization
- `3ce7666` test: verify Autoencoder detection with visualizer

## 2026-03-24（文献调研与图表规划）

### 子刊文献调研完成
- **报告**：`docs/LITERATURE_REVIEW_VISUALIZATION.md`
- **文献数**：200 篇（Nature/Science/Cell 子刊 2023-2026）
- **图表记录**：234 条
- **三大任务文献**：DeepPredict 36篇 / DeepClassify 45篇 / DeepDetect 36篇
- **高频图表**：Confusion Matrix 18%、时序预测图 16.3%、ROC-AUC 12%
- **Git**：`36140cb`

### 各模块应补充的图表
- **DeepPredict**：相关热力图、误差分布、滚动预测图
- **DeepClassify**：SHAP Decision Plot、信号标注图、特征热图
- **DeepDetect**：异常区间可视化、PR曲线、异常类型分类图

## 2026-03-24（团队重组）

### 团队架构更新（身份固定原则生效）
- **主Agent**：角色转型为项目经理人（整合三模块 + 与项目秘书沟通）
- **DeepPredict-Agent**：`91758ccc-a685-4a2b-a383-27a1bdbd78e4` — 全职维护 DeepPredict
- **DeepClassify-Agent**：`8c9d6116-48b9-485d-98c4-c517f8ae895a` — 全职维护 DeepClassify
- **DeepDetect-Agent**：`b5f67e99-9f50-422a-a900-35a792a4debd` — 全职维护 DeepDetect
- **项目秘书**：`98ae4c56-447c-4219-aec3-03a7b9974a12` — 定时汇报 + 报告生成

### Agent 记忆文件（已固定）
- `DeepPredict/AGENT_MEMORY.md` ✅
- `DeepClassify/AGENT_MEMORY.md` ✅
- `DeepDetect/AGENT_MEMORY.md` ✅
- `PROJECT_SECRETARY_MEMORY.md` ✅

### Git 提交
- `02b5dd4` Update: AGENTS.md - 主Agent转型为项目经理人
- `917e1ee` Add: 项目开发报告 20260324-2

## 2026-03-24（问题修复与优化）

### torch 升级成功（DeepDetect-Agent）
- 操作：`pip install torch==2.5.0+cpu --index-url https://download.pytorch.org/whl/cpu`
- 结果：torch 2.5.0+cpu 安装成功，DLL加载正常，无需安装VC++ Redistributable
- numpy 1.26.4 依然兼容
- Commit：`8056bc1`

### Autoencoder 优化（DeepDetect-Agent）
- Early Stopping：patience=5, min_delta=1e-5
- Reduce max_epochs：200 → 50
- Batch Normalization：Encoder/Decoder 每层加入 BatchNorm1d
- ReduceLROnPlateau：mode='min', factor=0.5, patience=3, min_lr=1e-6
- Commit：`8056bc1`

### LSTM 参数优化（主Agent）
- 优化测试：seq_len/num_layers/epochs 多组合搜索
- 最优配置：seq_len=90, hidden_channels=64, num_layers=3, kernel_size=5, epochs=80
- 优化结果：R²=0.6258 ✅（超过0.55目标）
- 优化前：R²=0.545
- 更新到网站默认参数

### Git 提交
- `ee29395` Update: LSTM parameters optimized (seq_len=90, num_layers=3, epochs=80)
- `8056bc1` DeepDetect: torch 2.5.0 upgrade + Autoencoder efficiency optimization

## 2026-03-24（v1.1 最终确认）

### 各模块最终版本
- DeepPredict: v1.1（调试后确认 LSTM/RF/GB PASS）
- DeepClassify: v1.1（K-Fold + SHAP + 模型对比）
- DeepDetect: v1.1（LSTM检测器 + 滑动窗口 + 7种检测器）

### 已知问题
| 问题 | 影响 | 状态 |
|------|------|------|
| torch 2.11.0+ DLL 加载失败 | 升级torch受阻 | 稳定版：torch 2.2.0+cpu |
| CNN1D R²=0.449 | DeepPredict | 中（epoch/seq_len不足） |
| PatchTST 完整测试未跑 | DeepPredict | 低优先级 |

### 下一步计划
1. CNN1D 参数调优（DeepPredict）
2. 升级 torch 到 2.5+
3. 多变量支持（DeepDetect）
4. FastAPI 服务化

---

*最后更新：2026-03-24 10:46 GMT+8*
