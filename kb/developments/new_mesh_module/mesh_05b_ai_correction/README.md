# FEALPy | mesh_05b_ai_correction - 任务导航

- **任务编号**：mesh_05b_ai_correction
- **任务标题**：新网格模块 AI 修正：接口语义统一、基本算例跑通与旧类型依赖识别
- **版本**：v0.1
- **状态**：草案
- **创建时间**：2026-06-09
- **来源**：2026-06-09 小组周例会 T7 / 行动项 A5

## 快速导航

1. **[Task Brief](mesh_05b_ai_correction_task_brief.md)**
   - 定义任务来源、定位、边界、执行子任务、目标资产和完成判据。

2. **[AI Agent Prompt](mesh_05b_ai_correction_agent_prompt.md)**
   - 可直接交给 AI Agent 的中文执行提示。

3. **[Validation Checklist](mesh_05b_ai_correction_validation_checklist.md)**
   - 用于人工或二阶段 AI Review 的验收清单。

## 建议使用路径

- 任务负责人：先读 Task Brief，再按 Validation Checklist 检查输出。
- AI Agent：直接读取 Agent Prompt，但必须同时打开 Task Brief 中列出的文档与代码。
- 验收者：重点检查是否满足 A5：常用接口可用、基本简单算例跑通、接口约定清楚、旧类型依赖已识别。

## 关联资产

- 会议纪要：`D:/suanhai-repo/xihe/kb/meetings/2026/06/2026_06_09_group_weekly_meeting/meeting_notes.md`
- 总任务：`kb/developments/new_mesh_module/new_mesh_module_task_brief.md`
- 前置接口资产：`kb/developments/new_mesh_module/mesh_03_optimize_architecture/mesh_module_interface.md`
- 算法迁移资产：`kb/developments/new_mesh_module/mesh_05_algorithm_migration/`
- 代码目录：`fealpy/mesh/schema/`
- 测试目录：`tests/mesh/unit/schema/`
