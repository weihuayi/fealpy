# mesh_07_module_validation

新网格模块合并到 `develop` 后的团队研究工作验证任务。

## 任务目标

由每位团队成员检查自己是否有基于 FEALPy 的现有研究工作：

- 有：在统一 `develop` 提交上运行至少一个代表性端到端用例，记录结果并报告错误、Bug、迁移遗漏和风险；
- 无：提交不适用声明，不为满足形式要求虚构用例。

本任务收集并复核问题，为实现修复、回归补测和 `mesh_08` 最终验收提供输入；不在本任务内默认完成全部修复或最终验收。

## 已启用资产

| 资产 | 用途 |
|---|---|
| `mesh_07_module_validation_task_requirement.md` | 定义需求、约束、成功条件和待裁决事项 |
| `mesh_07_module_validation_task_brief.md` | 稳定任务定位、目标、输入输出和完成判据 |
| `mesh_07_module_validation_task_object_boundary.md` | 区分验证、研究工作、Bug 修复、补测和最终验收 |
| `mesh_07_module_validation_task_target_asset_design.md` | 统一成员记录、缺陷材料、台账和摘要的职责与路径 |
| `mesh_07_module_validation_task_verification_and_validation_design.md` | 区分实现正确性 Verification 与研究用途 Validation |
| `mesh_07_module_validation_task_execution_workflow.md` | 组织多人并行执行、复现、去重、复核和回流 |
| `mesh_07_module_validation_member_record_template.md` | 每位成员复制使用的记录模板 |
| `mesh_07_module_validation_defect_ledger.md` | 汇总唯一缺陷、复核状态和承接关系 |

## 未启用模板

- **Task Network Design**：父任务 `new_mesh_module_task_network_design.md` 已明确 `05/06 -> 07 -> 08` 及 07 向 05/06 的回流关系，本任务暂不重复建网。
- **Task Agent Prompt**：任务主位是团队成员执行自己的真实研究工作，不需要统一 AI Agent 自主执行提示词。
- **Task Closure**：当前只建立任务设计与执行入口，待实际验证、复核和结果摘要形成后再判断是否启用。

## 执行入口

1. 负责人在缺陷台账中填写 `develop` 完整 commit、验证窗口、成员数和复核责任。
2. 每位成员复制 `mesh_07_module_validation_member_record_template.md` 为 `records/<member_slug>_validation_record.md`。
3. 按 `mesh_07_module_validation_task_execution_workflow.md` 完成用例执行和问题报告。
4. 汇总人将问题去重后录入 `mesh_07_module_validation_defect_ledger.md`，为可行动问题分配 `M07-###` 编号。
5. 经人工复核后形成 `mesh_07_module_validation_result_summary.md`，进入修复、补测和 `mesh_08` 接口。
