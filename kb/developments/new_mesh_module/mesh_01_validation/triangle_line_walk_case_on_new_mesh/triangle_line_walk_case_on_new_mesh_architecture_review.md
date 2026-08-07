# Suanhai | Architecture Review | 新网格模块架构理解与问题清单

- **版本**：v0.1
- **状态**：草案
- **入库位置**：`kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/triangle_line_walk_case_on_new_mesh_architecture_review.md`
- **适用范围**：`mesh_01_validation` 下“三角形 Line Walk 算例验证”子任务

本文面向当前子任务语境，记录对新 `fealpy/mesh` 模块的架构理解、与设计文档的对齐情况，以及会直接影响三角形 Line Walk 点定位算例实现的问题清单。

## 一、阅读对象与判断边界

本轮阅读覆盖：

- `kb/design/mesh/mesh-data-structure.md`
- `kb/implementation/mesh/construct-pipeline.md`
- `kb/implementation/mesh/boundary-info-inferrence.md`
- `fealpy/mesh/schema`
- `fealpy/mesh/storage`
- `fealpy/mesh/topology`
- `fealpy/mesh/view`
- `mesh_01_validation` 总任务说明与当前 Line Walk 子任务说明

判断边界遵循当前子任务定位：本报告不做新网格模块全量审计，也不提出大规模重构方案，只记录会影响“基于新模块完成三角形 Line Walk 算例与最小验证”的架构理解、可用能力与问题。

## 二、架构理解

### 2.1 总体分层

设计文档给出的核心思想是“规则、事实、操作分离”：

- `schema`：描述实体类型的规则，例如拓扑维数、局部子实体模板、几何操作。
- `storage`：保存具体网格事实，例如节点坐标、实体连接关系、实体间 relation。
- `topology`：负责从高维实体构造低维实体、推断跨层关系、推断边界信息。
- `view`：提供用户访问入口，避免用户直接操作底层 storage。

当前代码基本延续了这个方向，但实际命名与文档已有差异：

- 文档中常用 `MeshStorage / EntityBlock / blocks`。
- 当前代码实现为 `MeshBlock / EntitySector / sectors`，见 `fealpy/mesh/storage/mesh_storage.py`。

这不是致命问题，但会显著增加新成员阅读文档和代码时的映射成本。

### 2.2 `schema` 层

`EntitySchema` 定义统一接口，`ShapedEntitySchema` 给出部分默认实现。各实体类型通过 `SCHEMA_REGISTRY` 注册，例如 `tri`、`edge`、`node`。

对当前 Line Walk 子任务最关键的是：

- `TriangleSchema.name = "tri"`，`top_dim = 2`。
- `TriangleSchema.local_faces = {"edge": [[0, 1], [0, 2], [1, 2]]}`。
- `TriangleSchema` 已实现 `barycenter`、`grad_lambda`、`measure`、`normal` 的一部分。
- `EdgeSchema` 已实现 `barycenter`、`grad_lambda`、`measure`、`tangent`。
- `NodeSchema` 只实现了 `barycenter`。

也就是说，三角形网格的基本实体规则已经存在，但部分几何 API 在二维三角形场景下还不能直接认为稳定可用。

### 2.3 `storage` 层

`MeshBlock` 是当前事实容器，主要字段为：

- `positions`：节点坐标。
- `sectors`：按 schema 名称保存实体连接关系。
- `relations`：按 `(src_name, tgt_name)` 保存实体关系。
- `root_entity_names`：记录最高维或构造入口实体。
- `_cache_boundary_info`：边界信息缓存。

对一个简单三角形网格，合理使用路径是：

1. 构造 `positions`。
2. 构造 `EntitySector("tri", cell)` 并作为 root 加入 `MeshBlock`。
3. 调用 `TopologyBuilder.construct(block)`。
4. 得到 `tri`、`edge`、`node` 三类 sector，以及 `("tri", "edge")`、`("edge", "node")` 两类 relation。
5. 用 `Mesh(block)` 和 `EntityView` 查询实体与关系。

这个路径可以支持当前算例的网格装配起点。

### 2.4 `topology` 层

`TopologyBuilder.construct` 从 root sector 出发，逐层根据 `local_faces` 构造低维实体。核心链路与 `construct-pipeline.md` 基本一致：

- 通过 `get_total_face` 生成候选低维实体。
- 通过 `_unique_unordered_rows_across` 对无向实体做去重。
- 写入低维 `EntitySector`。
- 写入源实体到低维实体的 `Relation`。

`TopologyInferer` 能基于已有相邻层级关系推断更低维关系，例如从 `tri -> edge` 和 `edge -> node` 推断 `tri -> node`。当前推断方向只支持从高维到低维。

`BoundaryInferencer` 基于 relation 做边界推断：

- 次高维实体通过被最高维实体引用次数判定边界。
- 最高维实体通过是否引用边界次高维实体判定边界。
- 更低维实体通过边界闭包向下传播。

这与 `boundary-info-inferrence.md` 的设计基本一致。

### 2.5 `view` 层

`Mesh` 是较薄的入口：

- `Mesh.sector(name)` 返回某类实体的 `EntityView`。
- `Mesh.sectors(top_dim)` 按拓扑维数枚举实体视图。
- `Mesh.top_dimension()` 从 root sectors 推断最高拓扑维数。

`EntityView` 提供用户 API：

- `indices`
- `size`
- `barycenter`
- `measure`
- `grad_lambda`
- `normal`
- `tangent`
- `boundary`
- `to(target)`

对当前算例而言，`mesh.sector("tri").indices`、`mesh.sector("tri").to("edge").tgt_indices`、`mesh.sector("edge").boundary()` 是最直接可用的接口。

`Mesh.legacy()` 提供旧 FEALPy 风格接口，但当前子任务要求主流程走新模块，因此它只能作为对照或临时辅助理解，不宜成为算例主入口。

## 三、当前模块对三角形 Line Walk 的支撑情况

三角形 Line Walk 需要的最低能力包括：

- 三角形顶点坐标访问。
- 单元到边的局部关系。
- 边到相邻单元的反向邻接，或等价的 cell-to-cell 邻接。
- 边界边识别。
- 目标点相对当前三角形各边的定向面积或等价符号判断。
- 能根据“穿过哪条边”找到下一个三角形。

当前新模块已经直接提供：

- 三角形实体存储：`EntitySector("tri", cell)`。
- 节点坐标存储：`MeshBlock.positions`。
- 低维实体构造：`TopologyBuilder.construct`。
- `tri -> edge` 关系：`mesh.sector("tri").to("edge").tgt_indices`。
- `edge -> node` 关系。
- 边界边判定：`mesh.sector("edge").boundary()`。
- 三角形重心与梯度类接口的一部分。

当前仍需要当前子任务补充或绕开的小能力：

- 从 `tri -> edge` 反推 `edge -> tri` 或 `cell_to_cell` 邻接。
- 将 Line Walk 的“局部边编号”与当前 `TriangleSchema.local_faces` 顺序明确对齐。
- 对二维三角形使用可靠的有向面积计算，而不是依赖当前 `TriangleSchema.measure()`。
- 对起始单元、终止条件、边界穿出和路径记录做算例级封装。

因此，当前新模块足以作为算例主入口，但还不足以“零补充”完成 Line Walk。最小可接受做法是在算例中显式声明：网格构造、实体查询、`tri -> edge` 与边界判定由新模块提供；`edge -> tri / cell neighbor` 作为当前算例辅助逻辑由 `tri -> edge` 派生。

## 四、问题清单

### P1. 缺少直接支撑 Line Walk 的反向邻接或 cell-to-cell 查询

- **位置**：`EntityView.to()` 只委托 `schema.relation()`；`ShapedEntitySchema.relation()` 在缺 relation 时调用 `TopologyInferer.infer()`；`TopologyInferer.infer()` 要求 `src_dim > dst_dim`。
- **表现**：当前可以自然查询 `tri -> edge`，但不能查询 `edge -> tri`，也没有 `tri -> tri` 或 `cell_to_cell` API。
- **影响**：Line Walk 的核心步骤是“穿过当前三角形的一条边，进入共享该边的相邻三角形”。没有反向邻接时，算例必须手写从 `tri -> edge` 到 `edge -> tri` 的辅助表。
- **建议方向**：当前子任务可先实现最小局部辅助函数，例如由 `c2e` 构造 `edge_to_cells`。后续模块层面建议增加 relation 反向推断或邻接查询 API，并明确其 relation 表达形式。

### P2. 二维三角形 `measure()` / `normal()` 当前不可直接使用

- **位置**：`fealpy/mesh/schema/triangle.py` 中 `measure()` 与 `normal()` 使用 `bm.linalg.cross(v1, v2)`。
- **表现**：在二维坐标 `positions.shape[1] == 2` 的三角形网格上，`bm.linalg.cross` 会报错，原因是输入向量不是三维向量。
- **影响**：三角形算例通常是在二维 box 区域上运行。当前几何量接口不能作为二维三角形面积或定向面积判断的可靠基础。
- **建议方向**：当前 Line Walk 算例应使用显式二维有向面积公式。后续模块层面应让 `TriangleSchema.measure()` 对 `GD == 2` 使用行列式面积，对 `GD == 3` 使用叉乘范数。

### P3. `TriangleSchema.local_faces` 的局部边语义不利于 Line Walk

- **位置**：`TriangleSchema.local_faces = {"edge": [[0, 1], [0, 2], [1, 2]]}`。
- **表现**：该顺序不是常见的“与顶点 0/1/2 相对的边”顺序，也没有方向或 opposite-vertex 元数据。
- **影响**：Line Walk 常根据某个 barycentric 坐标或有向面积符号确定穿过哪条对边。如果局部边编号没有明确语义，就必须在算例中额外维护“负符号位置 -> 当前 local edge -> 全局 edge”的映射。
- **建议方向**：当前算例应显式记录使用的局部边顺序，并用节点集合匹配而不是假设 opposite-vertex 顺序。后续模块可考虑为 simplex schema 增加稳定的 `local_edges_opposite_vertex`、方向约定或 local relation 元数据。

### P4. 边界缓存没有失效机制

- **位置**：`MeshBlock._cache_boundary_info` 存储在 `mesh_storage.py`；`ShapedEntitySchema.boundary()` 首次调用后缓存 `BoundaryInferencer.infer_all()` 的结果。
- **表现**：`MeshBlock.add_sector()`、`TopologyBuilder.construct()` 会修改 sectors 与 relations，但没有清空 `_cache_boundary_info`。
- **影响**：如果在构造或修改 relation 之前调用过 `boundary()`，之后再查询边界可能得到过期结果。当前 Line Walk 算例一般会先构造后查询，因此可规避；但这属于新模块稳定化验证中的缓存一致性风险。
- **建议方向**：在 `add_sector()`、relation 写入、sector indices 覆盖等路径上统一失效边界缓存。当前算例文档中应约束“拓扑构造完成后再查询边界”。

### P5. 文档与代码命名不一致

- **位置**：设计文档使用 `MeshStorage / EntityBlock / blocks`；代码使用 `MeshBlock / EntitySector / sectors`。
- **表现**：同一概念需要人工映射。
- **影响**：新成员执行 `mesh_01_validation` 时容易误判代码位置或对象边界，尤其是在审查报告和后续算例说明之间造成术语漂移。
- **建议方向**：短期在本子任务资产中统一采用代码当前命名，并首次出现时注明与设计文档术语的对应关系。后续更新设计文档或增加“术语映射表”。

### P6. `EntitySchema` 接口面大于当前实现能力

- **位置**：`EntitySchema` 声明了 `barycentric`、`integral`、`transform`、`multi_index` 等接口；多个具体 schema 没有实现这些方法。
- **表现**：`EntityView` 直接暴露这些 API，但调用时可能落到 `NotImplementedError`。
- **影响**：对当前 Line Walk 而言不是主阻塞，但会影响“新模块 API 已完整可用”的判断。算例如果误用 `barycentric()` 或 `transform()`，会暴露接口断层。
- **建议方向**：当前算例只使用已验证可用的接口。后续模块应区分“稳定 API”和“预留 API”，或者在 view 层提供能力检测与更清晰错误信息。

### P7. `TopologyBuilder.construct()` 会覆盖同名 sector，且以 schema 名称作为唯一 key

- **位置**：`MeshBlock.sectors: dict[str, EntitySector]`，`TopologyBuilder.construct()` 中同名 sector 已存在时直接覆盖 `indices`。
- **表现**：同一 schema 名称只能有一个 sector。多个同类型 root block 或需要分块保留来源的情况无法表达。
- **影响**：当前均匀三角形网格只有一个 `tri` sector，不受影响。但这与混合网格或分块网格的长期扩展边界有关。
- **建议方向**：当前子任务无需处理。后续若要支持多块同类型实体，需要区分 schema type 与 block identity。

### P7. 边界推断没有显式暴露非流形状态

- **位置**：`BoundaryInferencer.infer_codim1()` 用引用计数生成 `mask = count == 1`，并把 `count` 放入 `BoundaryInfo`。
- **表现**：`count > 2` 的非流形实体不会被单独标记，只能由调用方检查 `count`。
- **影响**：当前规则三角形网格不受影响。若 Line Walk 用于非流形或错误网格，路径选择可能不唯一，而当前 API 不会直接提醒。
- **建议方向**：当前验证网格保持流形。后续可增加 `nonmanifold_mask` 或诊断接口。

### P9. Legacy 视图容易与“新模块主流程”边界混淆

- **位置**：`fealpy/mesh/view/fealpy_legacy.py`。
- **表现**：`FEALPyMesh` 提供旧接口风格，并包含一些四面体特化逻辑，例如 `localEdge`、`localFace`。
- **影响**：它能降低旧代码迁移门槛，但当前子任务要求主流程走新模块。如果算例大量依赖 legacy API，验证结论会变得不清楚。
- **建议方向**：当前 Line Walk 算例主流程使用 `Mesh`、`EntityView`、`MeshBlock`、`TopologyBuilder`。legacy 只作为对照，不作为核心依赖。

## 五、对当前 Line Walk 算例的执行建议

1. 实验网格优先采用当前子任务指定的 box 区域均匀三角形网格。构造方式是手动生成 `positions` 和 `tri` connectivity，再交给新模块构造低维实体。
2. 主流程入口使用 `MeshBlock`、`EntitySector("tri", ...)`、`TopologyBuilder.construct()`、`Mesh(block)`。
3. 算例中显式记录新模块直接提供的能力：`tri` 实体、`edge` 实体、`tri -> edge`、`edge -> node`、边界边、节点坐标访问。
4. 算例中显式记录辅助能力：由 `tri -> edge` 派生 `edge -> tri` 或 `cell_to_cell`，用于 Line Walk 步进。
5. 定向面积判断不要依赖当前 `TriangleSchema.measure()`，而应使用二维 determinant 公式。
6. 局部边映射不要假设 `TriangleSchema.local_faces` 是 opposite-vertex 顺序。应在算例中通过局部节点对与 `local_faces` 明确匹配。
7. 最小验证至少覆盖：内部点、靠近共享边的点、边界外穿出点、路径记录与边界终止。

## 六、阶段性判断

当前新网格模块的核心数据结构路线是成立的：用 `MeshBlock` 保存张量化事实，用 `EntitySector` 保存实体连接，用 `Relation` 保存层级关系，再由 `Mesh / EntityView` 提供访问入口，这条链路可以支撑简单三角形网格的构造、实体查询和边界推断。

对三角形 Line Walk 而言，当前模块已经具备“作为主入口”的基础可用性，但还缺少一个关键算法友好接口：反向邻接或 cell-to-cell 邻接。此外，二维三角形几何 API 的 `measure / normal` 问题会影响几何正确性验证，需要在算例中规避或后续修复。

因此，本子任务后续实现 Line Walk 时应采用“新模块主流程 + 最小辅助邻接派生”的策略，并把上述问题作为验证反馈沉淀给后续接口完善阶段。
