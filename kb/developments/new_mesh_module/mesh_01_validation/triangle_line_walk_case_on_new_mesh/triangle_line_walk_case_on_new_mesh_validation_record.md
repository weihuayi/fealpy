# Suanhai | Validation Record | 三角形 Line Walk 新网格模块算例验证记录

- **版本**：v0.1
- **状态**：草案
- **验证日期**：2026-05-06
- **适用范围**：`mesh_01_validation` 下“三角形 Line Walk 算例验证”子任务
- **对应设计**：`triangle_line_walk_case_on_new_mesh_algorithm_design_plan.md`

## 一、验证对象

本记录验证当前子任务中的三类资产：

- `fealpy/mesher/box.py` 中新增的 `Box2d` 二维 box 三角网格生成类。
- `examples/triangle_line_walk_on_new_mesh.py` 中的 Line Walk 算例实现。
- `tests/test_triangle_line_walk_on_new_mesh.py` 中的最小断言测试。

当前验证目标是功能正确性验证，不覆盖性能、复杂退化几何、非流形网格或通用点定位接口。

## 二、实验网格

- 网格区域：`[0, 1] x [0, 1]`
- 剖分参数：`nx = 10, ny = 10`
- 网格生成入口：`fealpy.mesher.box.Box2d.triangulate()`
- 节点数量：121
- 三角形数量：200
- 边数量：320
- 边界边数量：40

网格生成路径：

1. `Box2d.initialize()` 生成二维节点坐标和规则四边形连接关系。
2. `Box2d.triangulate()` 将每个四边形剖分为两个三角形。
3. `MeshBlock` 保存节点坐标。
4. `EntitySector("tri", tri)` 保存三角形实体。
5. `TopologyBuilder.construct()` 生成 `edge`、`node` 和关系表。

## 三、新模块参与度

新 `fealpy/mesh` 模块直接提供：

- `MeshBlock.positions`
- `EntitySector("tri", tri)`
- `TopologyBuilder.construct()`
- `Mesh` 与 `EntityView`
- `tri -> edge` relation
- `edge -> node` relation
- `edge.boundary().mask`

当前子任务补充的 mesher 能力：

- `Box2d.initialize()`
- `Box2d.triangulate()`

当前算例/测试程序补充的能力：

- `TriangleMeshData` 适配层
- `edge_to_cells` 派生
- `cell_neighbors` 派生
- `orient2d`
- `triangle_lambdas`
- Line Walk 主循环
- 路径记录对象
- brute-force oracle

## 四、运行命令与结果

算例运行命令：

```bash
python kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/examples/triangle_line_walk_on_new_mesh.py
```

输出摘要：

```text
nodes: 121
triangles: 200
edges: 320
point=[0.23, 0.37] status=on_edge cell=46 path=[0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 46]
point=[0.83, 0.71] status=inside cell=174 path=[0, 1, 20, 21, 40, 41, 42, 43, 62, 63, 64, 65, 84, 85, 86, 87, 106, 107, 108, 109, 128, 129, 130, 131, 150, 151, 152, 153, 172, 173, 174]
point=[-0.1, 0.5] status=outside cell=None path=[0, 1, 2, 3, 4]
```

注：运行时出现 matplotlib cache 目录不可写提示，该提示来自导入链中的 matplotlib 配置，不影响当前算例结果。

测试运行命令：

```bash
python -m pytest kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/tests/test_triangle_line_walk_on_new_mesh.py -q
```

测试结果：

```text
6 passed in 0.63s
```

## 五、断言覆盖

当前测试覆盖：

- `Box2d.triangulate()` 能生成新 `Mesh` 对象。
- 默认网格节点数为 121。
- 默认网格三角形数为 200。
- `tri_to_edge` 形状为 `(200, 3)`。
- `edge_to_node` 每条边有 2 个节点。
- 边界边数量为 40。
- 派生 `neighbors` 形状为 `(200, 3)`。
- 边界邻接槽位数量为 40。
- 每条边邻接 cell 数量为 1 或 2。
- 若干 cell 重心可被 Line Walk 定位到 brute-force oracle 给出的候选 cell。
- 外部点 `[-0.1, 0.5]` 被判定为 `outside`。
- 路径记录中每一步选择的穿越边等于最负 `lambda` 对应的边。
- Example 源码未直接导入或使用非 `bm` 数组库。

## 六、尚未覆盖

当前测试尚未覆盖：

- 多 backend 参数化验证。
- 非流形边处理。
- 退化三角形处理。
- 大规模网格性能。
- 所有边上点和顶点点归属策略。
- `TriangleSchema.measure()` 的二维修复。
- 通用 `cell_to_cell` 或反向 relation API。

## 七、暴露问题说明

### 7.1 二维三角形几何量接口需要设计确认

当前 `TriangleSchema.measure()` 与 `TriangleSchema.normal()` 通过三维叉乘计算几何量。这个实现适合三维嵌入三角形；但当前验证网格是二维 box 区域三角形，节点坐标形状为 `(Nnode, 2)`。

因此当前记录将它列为“需要设计确认”的问题，而不是直接判定为必须修复：

- 若设计目标只支持三维嵌入三角形，则应明确二维三角形不调用这些几何接口，或者由 mesher 生成三维坐标。
- 若设计目标支持二维平面三角形，则 `measure()` 应支持二维行列式面积公式，`normal()` 也应明确二维返回语义。

当前 Line Walk 算例没有调用 `TriangleSchema.measure()` 或 `TriangleSchema.normal()`。算例使用局部 `orient2d()` 与 `triangle_lambdas()` 完成二维符号判断，因此该问题没有阻塞当前点定位验证。

### 7.2 局部边顺序与 opposite-vertex 语义不一致

当前 `TriangleSchema.local_faces["edge"]` 的顺序为：

```python
[[0, 1], [0, 2], [1, 2]]
```

Line Walk 中局部重心坐标的边语义为：

```python
lambda0 -> edge (1, 2)
lambda1 -> edge (2, 0)
lambda2 -> edge (0, 1)
```

因此算法边顺序与 schema 边顺序存在明确差异。当前算例中显式构造：

```python
walk_to_schema_le = [2, 1, 0]
```

验证意义是：当某个 `lambda_i` 为最负值时，算法先找到它对应的 opposite-vertex 边，再映射到 schema 中实际的局部边编号，最后才能查询 `tri_to_edge[cell, schema_local_edge]`。如果省略这一步，Line Walk 会沿错误的全局边步进。

## 八、阶段判断

当前算例已经证明：新 `fealpy/mesh` 模块可以承担三角形网格事实存储、低维拓扑构造、关系查询、边界推断和视图访问；在测试程序补充反向邻接与二维几何判断后，可以支撑一个可运行、可解释、可断言验证的三角形 Line Walk 点定位算例。

当前算例同时暴露：反向邻接、`cell_to_cell` 查询、二维三角形几何量、局部边语义说明仍适合作为后续新网格模块接口完善候选。
