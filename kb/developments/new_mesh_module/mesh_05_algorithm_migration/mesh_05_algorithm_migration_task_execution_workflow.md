# FEALPy | Task Execution Workflow | 新网格模块算法实现迁移 执行工作流

- **版本**：v0.1
- **状态**：草案
- **入库位置**：`kb/developments/new_mesh_module/mesh_05_algorithm_migration/mesh_05_algorithm_migration_task_execution_workflow.md`
- **启用条件**：当需要定义当前 Task 的执行 How 时，本文件必须启用
- **适用范围**：新网格模块算法实现迁移的多人协作执行工作流

本文档用于定义当前 Task 的执行 How。

## 一、Workflow 定位

本 Workflow 以“迁移清单 → 并行迁移 → 验证收口”为主线，强调多人并行与可追溯交接，避免算法迁移变成不可控的散点修改。

## 二、Workflow Input

- 迁移优先级清单（来自任务负责人确认）
- Entity Schema 接口标准（来自 mesh_03 资产）
- 维度语义约定（normal / tangent）

## 三、Workflow Node 设计

### Node 1：迁移清单冻结

- 建立迁移映射表（旧实现 → 新 schema 方法）
- 标注优先级、负责人、验证方式

### Node 2：几何类算法迁移

- 迁移 `measure / barycenter / grad_lambda / normal / tangent`
- 按实体类型拆分并行执行

### Node 3：`multi_index` 迁移

- 迁移 `multi_index`，以 `fealpy/mesh/topology/ipoints.py` 的单纯形多重指标生成器为基础
- 明确返回形状与指标顺序

### Node 4：局部规则与 `ccw` 元数据迁移

- 迁移 `local_entity` 等局部索引规则
- 在每种 schema 内补齐 `ccw` 类属性并记录顺序约定

### Node 5：验证与收口

- 汇总验证脚本与测试
- 更新迁移映射表与风险记录

## 四、Workflow Gate 与 Transition 设计

- **Gate A**：迁移清单冻结后，才允许并行迁移 Node 2~4。
- **Gate B**：Node 2~4 完成后，进入 Node 5 验证收口。
- 若发现算法语义不清，回流 Node 1 补齐规则说明。

## 五、收口条件

- 迁移清单优先级项全部完成，且验证材料可复现。
- 迁移映射表与风险记录已更新并可追溯。

## 附录 A：本文件版本演进记录

- **v0.1**：
	- 变更人：AI Agent
	- 变更时间：2026-05-26
	- 变更摘要：
		- 首次建立新网格模块算法实现迁移 Task Execution Workflow
		- 明确五节点执行结构与回流规则
