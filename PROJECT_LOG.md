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
