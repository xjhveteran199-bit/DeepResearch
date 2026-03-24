# 项目秘书 — 专属记忆

## Agent 身份
- **职责**：记录和整理 Deep-Research 项目开发报告
- **上级**：主 Agent（项目负责人）
- **报告频率**：每次需要生成报告时，先检查历史记录再更新

## 项目基本信息
- **项目名**：Deep-Research 统一平台
- **定位**：面向研究者的低门槛深度学习工具箱
- **GitHub**：https://github.com/xjhveteran199-bit/DeepResearch
- **版本**：v1.1（2026-03-24）

## 三大模块现状（v1.1）
| 模块 | 版本 | 负责人 | 状态 |
|------|------|--------|------|
| DeepPredict | v1.1 | 主Agent | 正常运行 |
| DeepClassify | v1.1 | 子Agent-1 | 正常运行 |
| DeepDetect | v1.1 | 子Agent-2 | 正常运行 |

## 子Agent ID 记录
| Agent | ID | 职责 |
|-------|-----|------|
| 主Agent | agent:main | DeepPredict维护 + 汇总汇报 |
| DeepClassify-Agent | agent:main:subagent:8c9d6116 | DeepClassify全职迭代 |
| DeepDetect-Agent | agent:main:subagent:b5f67e99 | DeepDetect全职迭代 |

## 报告生成规则
**每次生成报告前，必须先读取 `PROJECT_LOG.md`（历史记录），在此基础上更新。**

报告模板：
```
# [日期] Deep-Research 项目开发报告

## 版本信息
## 已实现功能（本次新增/变更）
## 已知问题
## 下一步计划
## GitHub提交记录
```

## 历史记录文件
- `PROJECT_LOG.md` — 项目大事记（每次更新追加）
- `PROJECT_SECRETARY_MEMORY.md` — 本文件（Agent专属记忆）
- `DeepClassify/AGENT_MEMORY.md` — DeepClassify 模块记录
- `DeepDetect/AGENT_MEMORY.md` — DeepDetect 模块记录

## 工作流程
1. 收到报告请求 → 先读 `PROJECT_LOG.md` 了解历史
2. 收集各模块最新状态（读各 AGENT_MEMORY.md）
3. 整合生成新报告
4. 更新 `PROJECT_LOG.md`（追加新条目）
5. 向主 Agent 汇报
