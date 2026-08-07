# FEALPy | Task Object Boundary | 新网格模块架构优化 对象边界

- **版本**：v0.1
- **状态**：草案
- **入库位置**：`kb/developments/new_mesh_module/mesh_03_optimize_architecture/mesh_03_optimize_architecture_task_object_boundary.md`
- **启用条件**：当需要界定当前 Task 的对象边界时，本文件必须启用
- **适用范围**：新网格模块架构优化的直接建设范围与相邻对象分界

本文档用于界定当前 Task 直接建设什么、不直接建设什么，以及与相邻对象如何分界。

## 一、边界定位

本任务专注于新网格模块的接口设计与实现优化，在保持核心架构不变的前提下，通过补充接口、完善元数据、改进管理机制来提升模块的易用性与适配度。任务不涉及模块的重构或大规模架构变更。

## 二、纳入范围

本任务**直接建设**以下内容：

- `fealpy/mesh/schema/` 中各 schema 的元数据补充（opposite-vertex 映射、方向约定等）。
- `fealpy/mesh/schema/` 中几何接口的维数自适应（measure、normal 等）。
- `fealpy/mesh/storage/` 中的缓存失效机制（自动或手动触发）。
- `fealpy/mesh/topology/` 中的反向关系推断或邻接查询接口。
- `fealpy/mesh/view/` 中的便利查询 API（cell_to_cell_neighbors、on_boundary 等）。
- `fealpy/mesh/` 中支持形状+维数组合查询的入口接口。
- 更新或新增的单元测试、验证脚本。
- 设计文档与术语映射表的更新。
- 代码中的文档字符串与注释的补充与统一。

## 三、排除范围

本任务**不直接建设**以下内容：

- 新网格模块的核心架构重设计（schema / storage / view / topology 的职责重新划分）。
- 全仓库网格模块的全面迁移或替换。
- 性能优化与并行化改造（除必要的缓存优化外）。
- 新的 mesher 或网格算法实现（这些应在算例任务中推进）。
- 可视化、IO 或其他模块的改造。
- 对旧网格模块的修改或同步。

## 四、相邻对象分界

- **与 mesh_01_validation 的关系**：本任务是对其反馈的直接响应与实施，不再进行新的验证评估。
- **与后续算例任务的关系**：本任务提供优化后的接口，后续算例任务（如四面体质量计算）应基于这些新接口进行演示与验证。
- **与其他模块的关系**：本任务仅涉及 `fealpy/mesh/` 模块本身，不对求解器、mesher 或其他模块做需求。
- **与文档更新的关系**：`kb/design/mesh/` 与 `kb/implementation/mesh/` 的同步更新属于本任务一部分，确保代码与文档对齐。

## 五、待确认边界点

- 反向关系推断的实现位置：应该在 `TopologyInferer` 中扩展，还是作为 `Mesh / EntityView` 的便利包装？
- 元数据存储形式的最终决策：在 schema 属性、专用文件还是文档中？
- 缓存失效机制的触发时机：是隐式自动失效，还是显式 API 调用失效，还是两者结合？
- 便利 API 的完整范围与优先级顺序。

## 附录 A：本文件版本演进记录

- **v0.1**：
	- 变更人：AI Agent
	- 变更时间：2026-05-07
	- 变更摘要：
		- 首次建立新网格模块架构优化 Task Object Boundary
		- 明确纳入范围、排除范围与相邻对象分界
