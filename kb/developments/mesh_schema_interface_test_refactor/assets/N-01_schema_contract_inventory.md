# N-01 | Schema 规范盘点

- 任务：`mesh_schema_interface_test_refactor`
- 节点：N-01 规范盘点
- 版本：v0.1
- 状态：已形成 N-04 输入草案
- 规范基线：`kb/design/mesh/mesh_module_contract.md`
- 接口基线：`kb/design/mesh/mesh_entity_functions.md`
- 实现签名核对：`fealpy/mesh/schema/entity_schema.py`
- 交付对象：为 N-03 契约冻结和 N-04 case 设计提供字段级依据

## 1. 使用规则

1. 测试期望值首先来自本文件列出的设计约定；生产实现和旧测试只能用于发现偏差，不能反向修改期望值。
2. `OFace` 表示按有向局部实体关系组织的索引，`SFace` 表示无向/标准局部实体关系。测试数据必须同时保留两者（适用时）。
3. 每个 case 至少应能追溯到：形状、拓扑维数、顶点编号、局部实体编号、几何嵌入维数和支持接口。
4. 代码与规范不一致时保留失败证据，进入迁移/缺陷记录；不得将当前实现输出直接固化为契约。

## 2. 形状、顶点和局部实体基线

下表中的索引均为单个根实体内部的局部顶点编号，从 0 开始。

| Schema | 拓扑维数 | 顶点/几何族 | `OFace` 局部实体 | `SFace` 局部实体 | N-04 最低覆盖 |
|---|---:|---|---|---|---|
| `point` | 0 | `[0]` | 无 | 无 | 单点、嵌入几何维数 |
| `segment` | 1 | `[0, 1]` | point: `[[0], [1]]` | point: `[[0], [1]]` | 正向/反向 orientation、index |
| `tri` | 2 | `[0, 1, 2]`，0-1 为底边，2 为顶点 | segment: `[[1,2], [2,0], [0,1]]`；point: `[[0],[1],[2]]` | segment: `[[1,2], [0,2], [0,1]]`；point: `[[0],[1],[2]]` | 规则/非规则、2D/3D 嵌入 |
| `quad` | 2 | 图示顺序：底边 `0-1`，右上 `2`，左上 `3` | segment: `[[0,1], [1,2], [2,3], [3,0]]`；point: `[[0],[1],[2],[3]]` | segment: `[[0,1], [1,2], [2,3], [0,3]]`；point: `[[0],[1],[2],[3]]` | 规则/扭曲、2D/3D、环状顺序 |
| `tet` | 3 | `[0,1,2,3]`，底面 0-1-2，顶点 3 | tri: `[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]`；segment: `[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]`；point: 单点序列 | tri: `[[1,2,3],[0,2,3],[0,1,3],[0,1,2]]`；segment/point 同 OFace | 规则/非规则、3D、局部面方向 |
| `prism` | 3 | 底三角形 0-1-2、顶三角形 3-4-5 | tri: `[[0,2,1],[3,4,5]]`；quad: `[[0,1,4,3],[1,2,5,4],[0,3,5,2]]`；segment/point 为规范清单中的顺序 | tri: `[[0,1,2],[3,4,5]]`；quad: `[[0,1,3,4],[1,2,4,5],[0,2,3,5]]`；segment/point 与 OFace 相同 | 能力声明驱动；未适用接口显式排除 |
| `pyramid` | 3 | 底面 0-1-2-3，顶点 4 | tri: `[[0,1,4],[1,2,4],[2,3,4],[3,0,4]]`；quad: `[[0,3,2,1]]`；segment/point 为规范清单中的顺序 | tri: `[[0,1,4],[1,2,4],[2,3,4],[0,3,4]]`；quad: `[[0,1,2,3]]`；segment/point 与 OFace 相同 | 能力声明驱动；独立记录棱锥专属限制 |
| `hex` | 3 | 底层 0-1-2-3、上层 4-5-6-7；每层按规范图示编号 | quad: `[[0,3,2,1],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,4,7,3],[1,2,6,5]]`；segment/point 为规范清单中的顺序 | quad: `[[0,1,2,3],[4,5,6,7],[0,1,4,5],[2,3,6,7],[0,3,4,7],[1,2,5,6]]`；segment/point 与 OFace 相同 | 规则/扭曲、3D、顶点和局部面顺序 |

注意：上述完整 segment/point 列表以 `kb/design/mesh/mesh_module_contract.md` 为准。N-04 不应为相同的低维实体列表在每个形状模块中重复定义，应由 case/Schema 读取关系并在必要处做规范断言。

## 3. orientation 基线

| Schema | 规范状态 | 测试要求 |
|---|---|---|
| `point` | `[(0,)]` | 至少检查 identity orientation |
| `segment` | `(0,1)`、`(1,0)` | 检查局部朝向置换和 `global_permutations` |
| `tri` | 3 个循环排列 + 3 个反向排列 | 检查循环与反向集合，不得只检查数量 |
| `quad` | 4 个循环排列 + 4 个反向排列 | 检查环状顺序与反向顺序 |
| `tet/prism/pyramid/hex` | 本合同未为所有三维形状给出独立 `orientation` 表 | 在 N-03 标为待冻结；N-04 暂不猜测全排列，先覆盖 `OFace/SFace` 和可观察 permutation |

## 4. 标准 Schema 接口清单

接口来源为 `kb/design/mesh/mesh_entity_functions.md` 的标准函数表，基类签名以 `fealpy/mesh/schema/entity_schema.py` 为准。

### 4.1 拓扑

- `boundary(ctx) -> BoundaryInfo`：返回 `index` 和 `mask`；`mask` 长度等于当前目标实体数。
- `local_entity(tgt_name, indexing="o") -> list[list[int]]`：`indexing` 只能为 `"o"` 或 `"s"`。
- `relation(ctx, tgt_name) -> Relation`：关系方向为当前 schema 到目标实体。
- `size(ctx) -> int`：当前 sector 的实体数。

### 4.2 多重指标与朝向

- `multi_index(order, internal=False, tensorprod=True) -> Tensor`：行是多重指标，列对应局部顶点（或 `tensorprod=False` 时对应参考方向端点）；每行总阶数/各方向阶数应满足对应形状定义。
- `num_multi_index(order, internal=False) -> int`：必须等于 `multi_index(...).shape[0]`。
- `global_permutations(ctx, tgt_name, indexing="o") -> Tensor`：返回局部子实体到全局子实体的顶点置换。
- `vo_to_do(order) -> dict[tuple[int, ...], Tensor]`：返回 vertex orientation 到 DoF ordering 的映射。

### 4.3 几何、映射和积分

- `barycenter(ctx, index) -> Tensor`：通常为 `(NC, GD)`；`index` 子集必须只选择对应实体。
- `bc_to_point(ctx, bcs, index) -> Tensor`：物理点映射；输出前导维度应保留实体和积分点维度，张量积形状按各方向广播。
- `geo_dimension(ctx) -> int`：等于位置数组的最后一维。
- `grad_shape_function_barycentric(bcs, p) -> Tensor`：文档约定形状 `(NQ, num_shape, num_bc)`。
- `grad_shape_function_reference(bcs, p) -> Tensor`：文档约定形状 `(NQ, num_shape, ref_dim)`。
- `grad_shape_function_cartesian(ctx, bcs, p, index=None) -> Tensor`：由参考梯度和 Jacobian 变换得到；不能只检查数值，还要检查维度。
- `jacobi_matrix(ctx, bcs, index) -> Tensor`：文档约定 `(NC, NQ, GD, ref_dim)`。
- `measure(ctx, index) -> Tensor`：每个选中实体一个测度；节点测度为 0 的语义来自计算视图文档。
- `normal(ctx, index) -> Tensor`：返回未归一化法向/有向测度向量的具体维度须按 shape case 冻结。
- `tangent(ctx, index) -> Tensor`：返回切向量集合；具体数量随形状和嵌入维数变化。
- `integral(ctx, func, q, index) -> Tensor`：积分结果前导实体维度与 `measure` 一致。
- `quadrature_formula(q, qtype="legendre", device=None) -> Quadrature`：case 需检查点、权重和权重总和的形状。
- `shape_function(bcs, p) -> Tensor`：形函数值的最后一维为 `num_shape`，并检查分割统一性/节点插值性质（适用时）。

## 5. N-04 字段到规范的映射

| case 字段 | 必填 | 规范依据 | 设计约束 |
|---|---|---|---|
| `name` / `schema` | 是 | 形状格式与 Schema 注册 | 使用 `point/segment/tri/quad/tet/prism/pyramid/hex` |
| `positions` | 是 | 几何接口约定 | 固定数组；不依赖随机数；支持嵌入维数 |
| `indices` | 是 | `EntityContext`/sector 约定 | `(NC, NV)`；允许多实体和子集测试 |
| `geometry_kind` | 是 | 任务需求 | 至少 `regular`、`distorted` |
| `embedding` | 是 | `geo_dimension` | 记录拓扑维数与几何维数 |
| `supported_interfaces` | 是 | 标准函数表 | 能力筛选的唯一来源；不在测试函数内散落 shape 判断 |
| `expected_local_entities` | 适用 | `mesh_module_contract.md` | 同时表达 OFace/SFace；关系期望不得从实现复制 |
| `bcs` / `polynomial_order` | 按接口 | EntitySchema 签名 | 与参考实体匹配；固定、可复现 |
| `expected` / `reference` | 按接口 | 数学定义或独立公式 | 非仿射积分必须走独立参考计算 |
| `index_cases` | 是 | 接口 `index` 参数 | 至少 `None`、单项、切片/子集（适用时） |
| `backend` | 是 | 项目 backend 约定 | 明确支持、跳过或未验证状态 |

## 6. 规范偏差登记规则

N-04 看到以下情况时不得自行修订规范：

- 生产实现与 `OFace/SFace` 或顶点顺序冲突；
- 输出形状与基类 docstring 或接口文档冲突；
- prism/pyramid 的标准接口适用范围不明确；
- backend 对同一接口的行为不一致；
- 非仿射积分没有独立参考值。

记录字段至少包括：`deviation_id`、`source`、`case_id`、`interface`、`normative_expectation`、`observed_behavior`、`decision`、`owner`。未裁决项进入 N-03 冻结文档的开放问题表。

## 7. N-04 交付检查

- [ ] 每个 case 字段均有本文件或其引用规范依据。
- [ ] quad 环状顺序和 hex 顶点/局部面顺序有显式断言输入。
- [ ] regular/distorted、embedded/multi-entity/index-subset 均有 case 计划。
- [ ] 不适用接口通过能力声明排除，并保留原因。
- [ ] 不使用 `Path(__file__).parents[...]` 或 ad-hoc `sys.path` 导入数据。
- [ ] 规范偏差与未决契约不被写成正常期望值。
