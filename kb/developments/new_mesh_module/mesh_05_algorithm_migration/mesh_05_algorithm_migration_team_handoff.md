# FEALPy | 新网格模块算法实现迁移 - 协作交接说明

- **版本**：v0.1
- **状态**：草案
- **入库位置**：`kb/developments/new_mesh_module/mesh_05_algorithm_migration/mesh_05_algorithm_migration_team_handoff.md`
- **适用范围**：新网格模块算法迁移协作与团队分工

本文档用于向团队成员交代任务边界、接口标准与语义约定，避免算法迁移时出现口径不一致。

## 一、新网格模块基本架构（简版）

新网格模块分为三层：

1. **规则模板层（Entity Schema）**：定义实体规则与算法模板，**本任务只在此层迁移算法**。
2. **拓扑存储层（MeshBlock / EntitySector）**：保存离散事实，不承载算法。
3. **计算视图层（Mesh / EntityView）**：提供调用入口，不在本任务中增加算法实现。

当前任务的工作边界是：**只把算法迁移进 Entity Schema，并补充缺失算法**，不修改 view / storage / topology 的结构。

## 二、任务边界与协作分工提醒

- 迁移对象是算法实现，不是接口设计。
- 迁移输出必须遵循 Entity Schema 接口标准。
- 若算法依赖 view 或 storage 的额外状态，请记录为风险项，不要在本任务中扩展架构。
- 每种 schema 必须补齐 `ccw` 类属性，用于表达子实体的逆时针排列顺序。
- 迁移后需要留下可追溯映射与最小验证材料。
- 通过 AI 辅助产生单元测试文件。

==**！！特别提示！！**==

新网格数据结构的不同之处：按形状分配算法，而非按网格类型分配。因此：

- 不存在如“三角形网格”的概念，仅有“包含三角形的网格”这种说法；
- Schema 中的方法仅需对这种形状本身负责，**而不必考虑低维面的情况**，因为低维面自然会被分配到其对应的形状 Schema 上。

因此在迁移算法时，应把低维面情形的分支结构去除。

## 三、Entity Schema 接口标准（来自子任务 03）

### 3.1 实体拓扑方法

| 方法名                | 参数名           | 返回类型            | 用途                     |
| ------------------ | ------------- | --------------- | ---------------------- |
| ~~`boundary`~~     | ctx           | BoundaryInfo    | 获取该种形状的实体的边界信息。        |
| ~~`local_entity`~~ | tgt_name      | list[list[int]] | 获取用于构造局部子实体的索引。        |
| ~~`relation`~~     | ctx, tgt_name | Relation        | 获取该实体与另一种形状的实体之间的拓扑关系。 |
| ~~`size`~~         | ctx           | int             | 获取该种形状的实体的数量。          |

这部分已经由中间类 `ShapedEntitySchema` 实现，不写。

### 3.2 多重指标方法

| 方法名           | 参数名 | 返回类型   | 用途            |
| ------------- | --- | ------ | ------------- |
| `multi_index` | p   | Tensor | 计算该种实体上的多重指标。 |

`p` 参数类型一定是一个或多个整数构成的**元组**。

### 3.3 几何计算方法

| 方法名                  | 参数名             | 返回类型   | 用途             |
| -------------------- | --------------- | ------ | -------------- |
| `barycenter`         | ctx, index      | Tensor | 计算该种形状实体的重心。   |
| `bc_to_point`        | ctx, bcs, index | Tensor | 把重心坐标转化为笛卡尔坐标。 |
| `geo_dimension`      | ctx             | int    | 获取几何维数。        |
| `grad_lambda`        | ctx, index      | Tensor | 重心坐标对笛卡尔坐标的梯度。 |
| `quadrature_formula` | q, qtype        | 积分公式   | 在该种形状实体上的积分公式。 |
| `measure`            | ctx, index      | Tensor | 计算该种形状实体的测度。   |
| `normal`             | ctx, index      | Tensor | 计算该种形状实体的法向。   |
| `tangent`            | ctx, index      | Tensor | 计算该种形状实体的切向。   |

（1）`ctx` 参数类型为 `EntityContext`，可用于获取节点位置和实体到节点的索引：
	节点位置：`ctx.block.position`，相当于原来的 `node`；
	实体到节点的索引：`ctx.sector.indices`，相当于原来的 `cell`、`face`。

（2）`bcs` 参数类型一定是**张量构成的元组**：对于单纯形网格，元组中只有一个张量；其他形状如三棱柱，则包含多个张量。

（3）提醒：仅需考虑当前形状自身。比如在迁移 `quadrature_formula` 方法时，可省去 `etype` 参数及其分支结构。

### 3.4 类属性

| 属性名           | 类型                         | 含义               |
| ------------- | -------------------------- | ---------------- |
| `name`        | str                        | 该种形状的名称          |
| `top_dim`     | int                        | 该种形状的拓扑维数        |
| `local_faces` | dict[str, list[list[int]]] | 子实体的局部编号（多重指标相容） |
| `ccw`         | dict[str, list[list[int]]] | 子实体的局部编号（外法向相容）  |

现状：缺失 `ccw`，其它属性都有。

## 四、不同维度实体上的法向 / 切向定义准则

统一约定如下：

- 设拓扑维数为 **T**，几何维数为 **G**，则 **法向数量为 `G - T`，切向数量为 `T`**。
- 返回值类型 **必须是 Tensor**，形状为 `[实体数, 方向数, 几何维数]`，其中方向数允许为 0。
- 默认返回值不做单位化，以保持与 `measure()` 的一致性。

补充说明：

- 方向顺序由 schema 的局部顺序约定决定；`ccw` 与 `local_faces` 均可作为约定来源。
- 对于 `G - T = 0` 或 `T = 0` 的情况，对应方向数为 0，返回空方向维度的 Tensor。
- 若调用方需要单位法向或单位切向，应显式归一化。

## 五、`ccw` 类属性约定

- 每种 schema 必须添加 `ccw` 类属性，用于表达子实体的逆时针排列顺序。
- `ccw` 与 `local_faces` 可能不同，必须显式记录。
- 例：四面体的 `ccw` 约定可以写作：

```python
ccw = {
    "tri": [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
}
```

## 六、其他重要说明

- **算法放置纪律**：仅将算法放到 Entity Schema，禁止在 view 或 storage 层新增算法实现。
- **实现风格**：使用 `EntityContext`，避免引入状态或副作用，保持函数纯度。
- **迁移追溯**：每个算法都需要在迁移映射表中有来源与验证记录。
- **低成本验证**：每类算法至少具备一个可复现验证入口，避免“只有口头正确”。

## 七、任务分配表

按实体形状分配算法迁移任务：

| 形状  | Schema 所在文件（新）                      | 算法所在文件（旧）                                  | 开发人 |
| --- | ----------------------------------- | ------------------------------------------ | --- |
| 点   | fealpy/mesh/schema/node.py          | fealpy/mesh_old/node_mesh.py，但参考意义不大，建议直接写 | 李本桢 |
| 线段  | fealpy/mesh/schema/edge.py          | fealpy/mesh_old/edge_mesh.py，或者其它更高拓扑维数的网格 | 宋维豪 |
| 三角形 | fealpy/mesh/schema/triangle.py      | fealpy/mesh_old/triangle_mesh.py，或者四面体网格   | 戴俊  |
| 四边形 | fealpy/mesh/schema/quadrilateral.py | fealpy/mesh_old/quadrangle_mesh.py，或者六面体网格 | 陈春媚 |
| 四面体 | fealpy/mesh/schema/tetrahedron.py   | fealpy/mesh_old/tetrahedron_mesh.py        | 赵浩然 |
| 三棱柱 | fealpy/mesh/schema/prism.py         | fealpy/mesh_old/prism_mesh.py              | 钟吉祥 |
| 金字塔 | fealpy/mesh/schema/pyramid.py       | 不存在                                        | 胡凯  |
| 六面体 | fealpy/mesh/schema/hexahedron.py    | fealpy/mesh_old/hexahedron_mesh.py         | 高婷艺 |

算法所在文件，可到 `fealpy/mesh_old` 目录下的对应网格类型中查找，**部分方法可能在基类中**：
- fealpy/mesh_old/mesh_base.py
- fealpy/mesh_old/mesh_data_structure.py

测试文件放入 `tests/mesh/unit/schema` 目录内。
