# FEALPy | Task Network Design | 新网格模块算法实现迁移 任务网络设计

- **版本**：v0.1
- **状态**：草案
- **入库位置**：`kb/developments/new_mesh_module/mesh_05_algorithm_migration/mesh_05_algorithm_migration_task_network_design.md`
- **启用条件**：当需要识别当前 Task 的客观内在最优结构时，本文件必须启用
- **适用范围**：新网格模块算法实现迁移的协作结构与拆分认知

本文档用于识别、表达和压实当前 Task 的客观内在最优结构，并作为 Task Execution Workflow 的认知基础。

## 一、结构设计目标

在多人协作下，把“算法迁移”拆成可并行推进、可明确交付、可回溯验证的任务网络，避免把算法迁移变成不可控的散点改动。

## 二、核心组成单元

- **节点 A：迁移清单与映射建立**
  - 明确算法清单、优先级、旧实现位置与目标 Schema 方法。
- **节点 B：几何类算法迁移**
  - 迁移 `measure / barycenter / grad_lambda / normal / tangent` 等。
- **节点 C：多重指标 `multi_index` 迁移**
  - 迁移 `multi_index`，以 `fealpy/mesh/topology/ipoints.py` 的单纯形多重指标生成器为基础。
- **节点 D：局部索引与 `ccw` 规则迁移**
  - 迁移 `local_entity` 等局部拓扑规则，并补齐 schema `ccw` 元数据。
- **节点 E：验证与迁移记录收口**
  - 汇总验证材料，更新迁移映射表与风险记录。

## 三、关系结构

- A 是所有迁移工作的前置输入。
- B、C、D 可以并行推进，按实体类型或算法类别拆分。
- E 依赖 B、C、D 的迁移结果，并负责收口与可追溯性整理。

## 四、粒度与分界

- 每个节点应以“可交付的算法迁移结果”作为收口，而不是完成某个文件的修改。
- 并行拆分优先按“实体类型 + 算法类别”组合划分，避免多人同时修改同一 schema。
- 对缺少旧实现或语义不清的算法，应回流到节点 A 补齐清单与规则说明。

## 五、对 Execution Workflow 的前提输入

- 已冻结的算法迁移优先级清单。
- Entity Schema 接口标准的最终版本。
- `normal()` 与 `tangent()` 的维度语义判定结果。

## 附录 A：本文件版本演进记录

- **v0.1**：
	- 变更人：AI Agent
	- 变更时间：2026-05-26
	- 变更摘要：
		- 首次建立新网格模块算法实现迁移 Task Network Design
		- 采用 A~E 节点结构支持多人并行迁移
