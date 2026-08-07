# FEALPy | Task Brief | 新网格模块架构优化

- **版本**：v0.1
- **状态**：草案
- **入库位置**：`kb/developments/new_mesh_module/mesh_03_optimize_architecture/mesh_03_optimize_architecture_task_brief.md`
- **启用条件**：当需要为当前 Task 建立稳定本体表达时，本文件必须启用
- **适用范围**：`fealpy.mesh` 新网格模块架构优化、核心接口完善与用户友好性提升

本文档用于定义当前 Task 的稳定本体表达，回答该 Task 是什么、为什么成立、总体要解决什么问题，以及对象级完成判据为何。

## 一、Task 定位

本任务是对 [`mesh_01_validation`](kb/developments/new_mesh_module/mesh_01_validation/mesh_01_validation_task_brief) 验证阶段反馈的直接响应，目标是通过架构优化与接口完善，使新网格模块从"可用"升级到"好用、易用"。任务主位是提升模块在实际网格算法中的适配度与开发效率。

## 二、核心问题与主目标

验证阶段暴露了新网格模块在以下方面的不足：

1. **邻接查询能力不足**：缺少反向邻接或 cell-to-cell 查询，算法需手写派生逻辑。
2. **几何接口适配问题**：二维/三维几何量计算接口不自适应，导致维数不匹配。
3. **局部实体语义不清**：opposite-vertex 边序与其他形状的局部顺序缺乏统一的元数据说明。
4. **缓存管理脆弱**：边界信息缓存无失效机制，拓扑修改后容易产生不一致。
5. **接口管理不清**：EntitySchema 接口声明面过大，部分接口未实现，导致调用失败。
6. **文档与代码不同步**：术语映射（MeshStorage vs MeshBlock）增加理解成本。
7. **实体选取机制单一**：只能按 schema 名称选取实体，不能按形状与维数组合查询。
8. **便利 API 缺失**：view 层缺少常用的邻接、边界、梯度计算等便利查询接口。

本任务主目标是在不破坏核心架构前提下，通过接口设计、缓存管理、元数据补充与 API 扩展，使新网格模块能直接支撑常见网格算法，降低用户适配成本。

## 三、输入与输出预期

**输入预期：**

- `mesh_01_validation` 反馈的8个优化项与问题清单。
- 新网格模块当前代码与设计文档。
- 网格算法（Line Walk、质量计算等）对接口的实际需求。

**输出预期：**

- 8项架构优化的设计与实现方案。
- 补充或修改后的核心模块代码（schema / storage / topology / view）。
- 更新后的设计文档，术语映射表，元数据说明。
- 验证脚本与测试用例。
- 优化前后的对比说明与迁移指南。

## 四、对象级完成判据

本任务只有在同时满足以下条件时才算成立：

- **反向邻接查询**：`Mesh` 或 `EntityView` 提供可工作的反向关系推断或邻接查询接口，并通过测试。
- **几何量自适应**：`TriangleSchema` / `TetrahedronSchema` 等的 `measure()` 与 `normal()` 能正确处理二维/三维坐标，并有清晰的维数检查与文档。
- **opposite-vertex 元数据**：所有 simplex schema 中 `local_faces` 都有明确的 opposite-vertex 映射或方向约定，记录在代码文档或 schema 属性中。
- **缓存管理优化**：`MeshBlock` 或 `BoundaryInferencer` 实现缓存失效机制，关键修改路径自动清空 `_cache_boundary_info`。
- **接口管理完善**：`EntitySchema` 明确区分已实现 API 与预留 API，`EntityView` 调用时能检测能力并给出清晰错误信息。
- **术语统一**：设计文档更新为与代码一致的术语（MeshBlock, EntitySector 等），并提供术语映射表。
- **双重实体选取**：`Mesh` 支持按形状名称 + 维数的组合查询，如 `mesh.entity("triangle", 2)` 或 `mesh.entity(shape="triangle", dimension=2)`。
- **便利查询 API**：`EntityView` 或 `Mesh` 提供 `cell_to_cell_neighbors()`、`on_boundary()`、`gradient_indices()` 等常用接口。
- **验证资产齐全**：所有优化项都有对应的测试用例、验证脚本或文档说明，能证明优化有效且不破坏现有功能。
- **向后兼容**：现有 `mesh_01_validation` 中的算例仍能运行，不需大幅改写（除必要的 API 迁移外）。

## 五、待确认点

- 反向关系推断的完整边界：是只支持相邻层级（高维 → 低维），还是支持跨层级推断？
- 二维几何接口的完整设计：`normal()` 在二维场景下是否返回标量有向法向，还是其他语义？
- 元数据存储形式：是在 `EntitySchema` 属性中，还是单独的 `local_face_metadata` 文件？
- 缓存失效粒度：是全量失效还是支持部分刷新？
- 双重实体选取的 API 形式：是 `mesh.entity(name, dim)` 还是 `mesh.entity_by_shape_and_dim()`？

## 附录 A：本文件版本演进记录

- **v0.1**：
	- 变更人：AI Agent
	- 变更时间：2026-05-07
	- 变更摘要：
		- 首次建立新网格模块架构优化 Task Brief
		- 基于 mesh_01_validation 反馈的8项优化需求
		- 明确任务定位与完成判据
