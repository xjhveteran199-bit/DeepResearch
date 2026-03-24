# Deep-Research 多 Agent 协作架构

## 架构概览

```
项目经理人（主 Agent）
├── DeepPredict-Agent ──┐
│   (专职开发/优化/维护)   │
│   汇报频率: 按需主动汇报   │  ← 项目经理人汇总 → GitHub + 报告
├── DeepClassify-Agent ──┤
│   (专职开发/优化/维护)   │
│   汇报频率: 按需主动汇报   │
└── DeepDetect-Agent ────┘
    (专职开发/优化/维护)
    汇报频率: 按需主动汇报

项目秘书
└── 定时汇报（每2小时）→ 推送至 webchat
```

## 身份固定原则（重要）

**三个专职开发 Agent 的身份是永久固定的，不得随意重建或更换。**

每个开发 Agent 必须：
1. 持有独立的 `AGENT_MEMORY.md` 文件作为唯一记忆来源
2. 每次任务前读取自己的记忆文件，确保连续性
3. 每次任务后更新记忆文件，记录最新进展
4. 所有 Agent ID 和记忆文件路径写入 `AGENTS.md`，作为团队组织架构的官方记录

---

## Agent ID 固定清单

| Agent | Session ID | 记忆文件 | 负责模块 |
|-------|-----------|---------|---------|
| **主 Agent（项目经理人）** | agent:main | — | 整合协调 + 统一网站 |
| **DeepPredict-Agent** | `91758ccc-a685-4a2b-a383-27a1bdbd78e4` | `DeepPredict/AGENT_MEMORY.md` | DeepPredict |
| **DeepClassify-Agent** | `8c9d6116-48b9-485d-98c4-c517f8ae895a` | `DeepClassify/AGENT_MEMORY.md` | DeepClassify |
| **DeepDetect-Agent** | `b5f67e99-9f50-422a-a900-35a792a4debd` | `DeepDetect/AGENT_MEMORY.md` | DeepDetect |
| **项目秘书** | `98ae4c56-447c-4219-aec3-03a7b9974a12` | `PROJECT_SECRETARY_MEMORY.md` | 定时汇报 + 报告 |

---

## Agent 职责

### 项目经理人（主 Agent）
- 整合三个模块，协调子 Agent 工作
- 与项目秘书沟通，生成正式报告
- 维护 Deep-Research 统一网站
- 定期推送更新到 GitHub
- 接收子 Agent 汇报，决策优先级

### DeepPredict-Agent（专职开发）
- 路径: `C:\Users\XJH\DeepResearch\DeepPredict\`
- 记忆: `DeepPredict/AGENT_MEMORY.md`
- 职责: 时序预测模型的迭代开发与优化

### DeepClassify-Agent（专职开发）
- 路径: `C:\Users\XJH\DeepResearch\DeepClassify\`
- 记忆: `DeepClassify/AGENT_MEMORY.md`
- 职责: 信号分类模型的迭代开发与优化

### DeepDetect-Agent（专职开发）
- 路径: `C:\Users\XJH\DeepResearch\DeepDetect\`
- 记忆: `DeepDetect/AGENT_MEMORY.md`
- 职责: 异常检测方法的迭代开发与优化

### 项目秘书
- 记忆: `PROJECT_SECRETARY_MEMORY.md`
- 职责: 定时汇报（每2小时）+ 项目报告生成

---

## 汇报消息格式

### 开发 Agent → 项目经理人
```
【[模块名] 进展汇报】
时间: [时间戳]
本次更新: [具体内容]
修改文件: [文件名]
状态: 正常/有问题
下一步: [计划]
```

### 项目秘书 → webchat（定时）
```
## Deep-Research 定时汇报
时间: [时间]
DeepPredict： [状态/进展]
DeepClassify： [状态/进展]
DeepDetect：   [状态/进展]
网站：         [运行中/已停止]
GitHub：       [已同步/待推送]
```

---

## 版本管理
- 当前版本: v1.11 (2026-03-24)
- 升级规则: 主模块重大升级 → 主版本号变更；子模块更新 → 子版本号变更
- 版本文件: `VERSION.txt`
- 报告存档: `REPORTS/YYYYMMDD.md`

## 文档体系
| 文件 | 用途 |
|------|------|
| `AGENTS.md` | 团队组织架构（本文档） |
| `PROJECT_LOG.md` | 项目大事记（只追加不删除） |
| `PROJECT_SECRETARY_MEMORY.md` | 项目秘书专属记忆 |
| `[模块]/AGENT_MEMORY.md` | 各模块开发 Agent 专属记忆 |
| `REPORTS/YYYYMMDD.md` | 正式项目报告 |
