# DeepClassify Agent 专属记忆

## Agent 身份
- **职责**：全职负责 DeepClassify 模块的迭代更新
- **上级**：主 Agent（汇总 + 监控）
- **报告频率**：每 2 小时向主 Agent 汇报一次

## 模块基本信息
- **模块路径**：`C:\Users\XJH\DeepResearch\DeepClassify\`
- **版本**：v1.1（2026-03-24）
- **定位**：信号/数据分类，CNN1D/RF/GB/SVM 四种分类器
- **Git 远程**：`https://github.com/xjhveteran199-bit/DeepResearch`

## 当前版本功能
- CNN1D 分类器（PyTorch）：Conv1D×2 + BatchNorm + GlobalAvgPool
- RandomForest / GradientBoosting / SVM 分类器
- Accuracy / F1 / Confusion Matrix / ROC-AUC
- **新增**：K-Fold 交叉验证（StratifiedKFold，K=1~10可调）
- **新增**：SHAP 可解释性（TreeExplainer for RF/GB, KernelExplainer for CNN1D/SVM）
- **新增**：GB 后端选择（sklearn/LightGBM/XGBoost/auto）
- **新增**：模型对比功能（同时训练并比较所有模型）
- Web 界面：Gradio（端口7861独立运行）

## 技术栈
- Python 3.12 + PyTorch 2.2.0 + scikit-learn + Gradio 6.9.0
- numpy 1.26.4 + shap 0.51.0 + lightgbm 4.6.0
- **注意**：torch 2.2.0 需要 numpy < 2.0，shap 0.51.0 也兼容 numpy 1.26.4（无需强制升级）

## 环境说明（重要）
- torch 2.11.0+ 无法在此机器上加载 DLL（Windows 10 18362，VC++ runtime 14.14 太旧）
- torch 2.2.0+cpu 是目前可用版本，与 numpy 1.26.4 兼容
- SHAP 0.51.0 可与 numpy 1.26.4 共存（不需要 numpy>=2）
- **升级 torch 的方法**：关闭所有 Python 进程后，在新的 PowerShell 窗口执行：
  ```
  pip install torch==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu --force-reinstall
  ```

## 已知问题
- SHAP DeepExplainer（torch）不可用：torch 2.2.0 与 shap 0.51.0 的组合在当前环境测试 SHAP TreeExplainer 正常工作
- SHAP KernelExplainer 用于 CNN1D（慢但可用）：`shap.KernelExplainer(predict_fn, background)`

## 迭代 Roadmap
- [x] 解决 SHAP 与 torch 的 numpy 版本冲突（通过验证：shap 0.51.0 兼容 numpy 1.26.4）
- [x] 增加 LightGBM 作为 GB 分类器的后端选项
- [x] 增加 K-Fold 交叉验证
- [x] 增加模型对比功能
- [ ] 增加 SHAP beeswarm 决策图（可优化）
- [ ] CNN1D 小数据集稳定性改善（数据增强、dropout 调整）
- [ ] API 服务化（FastAPI）

## 汇报模板
每次汇报请包含：
```
## [时间] DeepClassify 进展汇报

### 本次更新
- 做了什么
- 修改了哪些文件

### 代码状态
- 语法检查：✅/❌
- 功能测试：✅/❌

### 下一步计划
```
