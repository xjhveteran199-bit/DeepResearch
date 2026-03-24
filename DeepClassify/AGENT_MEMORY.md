# DeepClassify Agent 专属记忆

## Agent 身份
- **职责**：全职负责 DeepClassify 模块的迭代更新
- **上级**：主 Agent（汇总 + 监控）
- **报告频率**：每 2 小时向主 Agent 汇报一次

## 模块基本信息
- **模块路径**：`C:\Users\XJH\DeepResearch\DeepClassify\`
- **版本**：v1.0（2026-03-24）
- **定位**：信号/数据分类，CNN1D/RF/GB/SVM 四种分类器
- **Git 远程**：`https://github.com/xjhveteran199-bit/DeepResearch`

## 当前版本功能
- CNN1D 分类器（PyTorch）：Conv1D×2 + BatchNorm + GlobalAvgPool
- RandomForest / GradientBoosting / SVM 分类器
- Accuracy / F1 / Confusion Matrix / ROC-AUC
- Web 界面：Gradio（端口7861独立运行）

## 技术栈
- Python 3.12 + PyTorch 2.2.0 + scikit-learn + Gradio 6.9.0
- numpy 1.26.4（注意：torch 2.2.0 需要 numpy < 2.0）

## 已知问题
- SHAP 分析依赖 shap 0.51.0，但 shap 0.51.0 要求 numpy>=2，与 torch 2.2.0 冲突
- CNN1D 的 torch 导入需在 numpy < 2 环境下

## 迭代 Roadmap
1. 解决 SHAP 与 torch 的 numpy 版本冲突（建议升级 torch 到 2.4+）
2. 增加多分类 ROC 曲线支持
3. 增加交叉验证（K-Fold CV）
4. 增加 SHAP 可解释性（解决版本冲突后）
5. 增加模型对比仪表盘
6. API 服务化（FastAPI）

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

### 问题/需要协助
```

## 最近更新记录
| 日期 | 更新内容 | 状态 |
|------|---------|------|
| 2026-03-24 | v1.0 初始版本，CNN1D/RF/GB/SVM 四种分类器 | ✅ 完成 |
