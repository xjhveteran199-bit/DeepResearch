# Deep-Research 多 Agent 协作架构

## 架构概览

```
主 Agent（我）
├── DeepPredict 维护（主 Agent 亲自负责）
├── DeepClassify 维护 ──┐
│  (子Agent-1 专属)     │
│  汇报频率: 每2小时一次  │  ← 主Agent汇总 → GitHub + 网站
└── DeepDetect 维护 ────┘
   (子Agent-2 专属)
   汇报频率: 每2小时一次
```

## Agent 职责

### 主 Agent
- 负责 DeepPredict 模块的迭代更新
- 汇总两个子 Agent 的汇报
- 维护 Deep-Research 统一网站
- 定期推送更新到 GitHub
- 监控子 Agent 状态（每2小时检查一次）

### 子Agent-1: DeepClassify 专属
- 路径: `C:\Users\XJH\DeepResearch\DeepClassify\`
- 记忆文件: `DeepClassify/AGENT_MEMORY.md`
- 职责: CNN1D/RF/GB/SVM 分类器的迭代开发
- 汇报: 每2小时向主 Agent 发送进展

### 子Agent-2: DeepDetect 专属
- 路径: `C:\Users\XJH\DeepResearch\DeepDetect\`
- 记忆文件: `DeepDetect/AGENT_MEMORY.md`
- 职责: 6种异常检测方法的迭代开发
- 汇报: 每2小时向主 Agent 发送进展

## 协作流程

1. **子Agent 定时汇报** → 发送 session message 给主 Agent
2. **主Agent 汇总** → 更新统一网站 + 推送 GitHub
3. **主Agent 监控** → 每2小时检查子Agent状态
4. **GitHub 同步** → 每次功能更新后自动 push

## 版本管理
- 当前版本: v1.01 (2026-03-24)
- 升级规则: 主模块重大升级 → 主版本号变更；子模块更新 → 子版本号变更
- CHANGELOG 位置: 每个模块目录下 CHANGELOG.md

## 汇报消息格式
```
【Deep-Research 协调汇报】
模块: [DeepClassify/DeepDetect]
时间: [时间戳]
状态: [正常/有问题]
进展: [具体内容]
问题: [如有]
下一步: [计划]
```
