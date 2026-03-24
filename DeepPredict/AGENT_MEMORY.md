# DeepPredict Agent 专属记忆

## Agent 身份
- **Session ID**：91758ccc-a685-4a2b-a383-27a1bdbd78e4
- **职责**：全职负责 DeepPredict 模块的迭代开发与优化
- **上级**：主 Agent（项目经理人）
- **报告频率**：按需主动汇报

## 模块基本信息
- **模块路径**：`C:\Users\XJH\DeepResearch\`（与 DeepClassify/DeepDetect 不同，DeepPredict 代码位于项目根目录，非子目录）
- **版本**：v1.11（2026-03-24）
- **定位**：时序回归预测，支持多种机器学习/深度学习模型
- **Git 远程**：`https://github.com/xjhveteran199-bit/DeepResearch`

## 代码结构
- **统一入口**：`deep_research_web.py`，端口 7862（Gradio Web UI）
- **预测模块**：`src/models/predictor.py`（Predictor 类）
- **可视化模块**：`src/visualizer.py`（PredictVisualizer 类）
- **SHAP 分析**：`src/utils/shap_analyzer.py`

## 当前版本功能（v1.11）
- **4种预测模型**：
  - RandomForest：R²=0.610, RMSE=2.55 ✅
  - GradientBoosting：R²=0.611, RMSE=2.55 ✅
  - LSTM：R²=0.6258（优化后参数：seq_len=90, hidden=64, layers=3, epochs=80）✅
  - CNN1D：R²=0.6538（优化后参数：seq_len=180, hidden=128, kernel=5, layers=3, epochs=100）✅
- **数据预处理**：自动归一化、缺失值处理、sliding window 构造
- **评估指标**：R²、RMSE、MAE 多指标
- **可视化**：PredictVisualizer → 预测时序图（真实值+预测值叠加 + 置信区间 + 局部放大子图）
- **SHAP 可解释性**：PredictVisualizer 集成 beeswarm summary plot（scienceplots 子刊风格）
- **BUG修复**：Gradio 6.x State subscript bug → 改用全局变量方案

## 技术栈
- Python 3.12 + PyTorch 2.2.0（CPU）
- numpy 1.26.4 + scikit-learn + Gradio 6.9.0
- shap 0.51.0 + matplotlib + scienceplots（子刊风格图表）

## 测试数据集
| 文件 | 大小 | 用途 |
|------|------|------|
| `dp_temperature.csv` | 67,921 bytes | 每日最低温度预测测试 |
| Airline Passengers（URL） | 144行 | 在线数据验证，R²=0.9648 ✅ |

## 最近 Git 提交
- `2211f11` feat: 集成 PredictVisualizer 可视化模块到统一平台
- `a5a822b` CNN1D参数优化: R²=0.6538 (seq=180, hidden=128, kernel=5, layers=3, epochs=100)
- `ee29395` Update: LSTM parameters optimized (seq_len=90, num_layers=3, epochs=80)
- `21a3388` Fix: Gradio State bug + port 7862

## 已知问题
| 问题 | 影响等级 | 状态 |
|------|---------|------|
| AGENT_MEMORY.md 此前不存在 | 中 | ✅ 2026-03-25 已创建 |
| PatchTST 完整测试未执行 | 低 | 暂缓 |
| CNN1D 对小数据集可能不稳定 | 中 | 已通过 Airline 数据集验证 R²=0.9648 |

## 下一步计划
1. 补充 PatchTST 完整测试
2. 增加更多可视化图表（误差分布、滚动预测、相关热力图）
3. 考虑独立 DeepPredict 子目录结构（与 DeepClassify/DeepDetect 对齐）

---

*最后更新：2026-03-25 07:46 GMT+8*
*创建者：主 Agent*
