# Suanhai | Summary | 三角形 Line Walk 新网格模块算例阶段摘要

- **版本**：v0.1
- **状态**：草案
- **日期**：2026-05-06
- **适用范围**：`mesh_01_validation` 下“三角形 Line Walk 算例验证”子任务

## 一、阶段产物

当前子任务已形成以下产物：

- 架构理解与问题清单：`triangle_line_walk_case_on_new_mesh_architecture_review.md`
- 算法设计规划：`triangle_line_walk_case_on_new_mesh_algorithm_design_plan.md`
- 二维 box 三角网格生成入口：`fealpy/mesher/box.py` 中的 `Box2d`
- Line Walk 算例：`examples/triangle_line_walk_on_new_mesh.py`
- 最小断言测试：`tests/test_triangle_line_walk_on_new_mesh.py`
- 验证记录：`triangle_line_walk_case_on_new_mesh_validation_record.md`

## 二、完成情况

当前实现采用“新模块主流程 + 算例内最小补充”的路线。

新模块承担：

- 网格节点坐标存储。
- 三角形实体存储。
- 边和节点等低维实体构造。
- `tri -> edge` 与 `edge -> node` 关系提供。
- 边界边推断。
- `Mesh / EntityView` 查询入口。

当前子任务补充：

- `Box2d` 二维 box 三角网格生成。
- `edge_to_cells` 与 `cell_neighbors` 派生。
- 二维 `orient2d` 和 `triangle_lambdas`。
- Line Walk 点定位主循环。
- brute-force oracle。
- 路径一致性验证。

## 三、验证结论

在默认 `[0, 1] x [0, 1]`、`nx = ny = 10` 的均匀三角形网格上：

- 节点数量为 121。
- 三角形数量为 200。
- 边数量为 320。
- 边界边数量为 40。
- 内部点、cell 重心点与外部点的定位结果通过断言验证。
- Line Walk 路径中每一步穿越边与最负局部重心坐标一致。

验证命令：

```bash
python -m pytest kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/tests/test_triangle_line_walk_on_new_mesh.py -q
```

结果：

```text
6 passed in 0.63s
```

## 四、暴露的问题

当前算例暴露出以下后续接口完善候选：

- 新 mesh 模块缺少直接的 `cell_to_cell` 邻接查询。
- 新 mesh 模块缺少从 `tri -> edge` 到 `edge -> tri` 的反向 relation 推断。
- `fealpy/mesher/__init__.py` 的旧 mesher 导入链依赖旧 `fealpy.mesh` 类，当前为使用新 `box.py` 入口做了兼容处理。

### 4.1 二维三角形几何量接口需要设计确认与单独验证

当前 `TriangleSchema.measure()` 和 `TriangleSchema.normal()` 的实现使用 `bm.linalg.cross(v1, v2)`。该实现天然适合嵌入三维空间的三角形，因为 `cross` 需要三维向量；但当前 Line Walk 算例使用的是二维 box 区域三角网格，节点坐标形状为 `(Nnode, 2)`，于是每条边向量是二维向量。

因此这里的问题不是简单地说“实现一定错了”，而是需要确认设计意图：

- 如果新 `TriangleSchema` 只打算支持三维嵌入三角形，那么二维 box 三角网格应在更高层转换为三维坐标，或文档中明确二维三角形不适用这些几何接口。
- 如果新 mesh 模块应支持二维平面三角形，那么 `measure()` 至少需要对 `GD == 2` 使用二维行列式面积公式，对 `GD == 3` 使用叉乘范数；`normal()` 也需要明确二维场景返回标量有向法向、二维旋转法向，还是不提供。

当前算例采用局部 `orient2d()` 和 `triangle_lambdas()` 规避这个问题，没有把 `TriangleSchema.measure()` 作为 Line Walk 的前置依赖。后续应补一个专门验证项：对二维三角形调用 `measure()` 的预期行为是什么，并据此补测试或修复实现。

### 4.2 `TriangleSchema.local_faces["edge"]` 与 Line Walk 的对边顺序不一致

当前 `TriangleSchema.local_faces["edge"]` 定义为：

```python
[[0, 1], [0, 2], [1, 2]]
```

但 Line Walk 使用局部重心坐标符号判断时，三个 `lambda` 的自然语义是“某个顶点的对边”：

```text
lambda0 -> opposite vertex 0 -> edge (1, 2)
lambda1 -> opposite vertex 1 -> edge (2, 0)
lambda2 -> opposite vertex 2 -> edge (0, 1)
```

也就是说，Line Walk 的算法边顺序是：

```python
[(1, 2), (2, 0), (0, 1)]
```

它与 schema 中的局部边顺序不是同一个顺序。直接对照可得映射：

```python
walk_to_schema_le = [2, 1, 0]
```

如果不显式映射，`lambda0 < 0` 时算法会错误地穿过 schema 的第 0 条边 `(0, 1)`，而正确含义应是穿过顶点 0 的对边 `(1, 2)`。当前算例通过 `build_walk_edge_to_schema_local_edge()` 对局部边做无向集合匹配，避免了把算法边编号和 schema 边编号混用。

## 五、阶段判断

三角形 Line Walk 算例已经证明：当前新 `fealpy/mesh` 模块具备支撑一个小型典型网格算法的基础能力，但仍需要算法侧补充反向邻接、局部几何判断和路径记录。

该结果足以作为 `mesh_01_validation` 的一条先验证据：新模块“能用”，但还没有把算法迁移常用的邻接和二维几何接口沉淀为稳定公共能力。
