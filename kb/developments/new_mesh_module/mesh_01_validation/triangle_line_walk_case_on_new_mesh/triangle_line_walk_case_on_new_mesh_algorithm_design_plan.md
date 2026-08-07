# Suanhai | Algorithm Design Plan | 基于新网格模块的三角形 Line Walk 算法设计规划

- **版本**：v0.2
- **状态**：修订稿
- **入库位置建议**：`kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/triangle_line_walk_case_on_new_mesh_algorithm_design_plan.md`
- **适用范围**：`mesh_01_validation` 下“三角形 Line Walk 算例验证”子任务
- **前置文档**：
  - `triangle_line_walk_case_on_new_mesh_task_brief.md`
  - `triangle_line_walk_case_on_new_mesh_task_object_boundary.md`
  - `triangle_line_walk_case_on_new_mesh_architecture_review.md`
  - `mesh-data-structure.md`

本文用于规划一个基于新 `fealpy/mesh` 模块运行的三角形 Line Walk 点定位算例。当前设计原则是：**主流程尽量走新网格模块；当前模块暂缺或尚未稳定暴露的能力，在算例或测试程序中做最小补充；最终验证材料必须明确区分“新模块直接提供的能力”“算例补充的能力”“后续适合沉入新网格模块的能力”。**

> 说明：本文是算法设计规划，不是仓库接口审计报告。凡涉及具体类名、方法名和返回字段的地方，后续实现时应以当前仓库实际代码为准；若实现接口与本文不同，应优先调整本文的“接口适配层”，而不是改变 Line Walk 算法主逻辑。

---

## 一、设计目标

当前算法设计服务于 `mesh_01_validation` 的“小而完整可用性验证”目标。该算例不追求通用点定位框架，也不建设完整 mesher 或高性能搜索系统，而是通过一个可控的三角形网格算法，验证新 `fealpy/mesh` 模块是否已经具备支撑典型网格算法的基础能力。

本算例需要达成以下目标：

1. 使用新 `fealpy/mesh` 模块构造或承载一个二维三角形网格。
2. 通过新模块入口访问节点坐标、三角形单元和必要的拓扑关系。
3. 在该网格上实现一个可运行的三角形 Line Walk 点定位过程。
4. 记录点定位路径，验证每一步步进方向与定向面积 / 重心坐标符号判断一致。
5. 明确当前新模块在算例中直接提供了哪些能力。
6. 明确哪些能力是在算例或测试程序中临时补充的，避免把补充逻辑误写成新模块既有能力。
7. 为后续是否需要把邻接构造、反向关系、二维几何量、局部边语义等能力沉入 `fealpy/mesh` 提供依据。

---

## 二、设计边界

### 2.1 纳入范围

当前设计纳入以下内容：

- 参考现有 `fealpy/mesher/box.py` 的新接口风格，补充 box 区域上的二维三角形网格生成类，作为缺省实验载体。
- 新 `fealpy/mesh` 模块的基本使用路径：网格事实存储、实体访问、关系访问、视图查询。
- 从三角形到边关系派生边到三角形关系，再派生三角形邻接关系的最小辅助逻辑。
- 二维定向面积与局部重心坐标符号判断。
- 单点 Line Walk 主流程。
- 多个代表性测试点的最小功能正确性验证。
- 路径记录、结果摘要和新模块参与度说明。

### 2.2 排除范围

当前设计不纳入以下内容：

- 大规模修改 `fealpy/mesh` 源码。
- 通用三角形网格生成器。
- 四边形、多边形、四面体或混合网格上的 Line Walk。
- 空间索引、全局搜索优化、多起点策略优化。
- 曲边单元、高阶网格或 CAD 约束下的点定位。
- 浮点退化情形的系统鲁棒性研究。
- 完整自动化测试体系或 CI 接入。

### 2.3 对“不修改源码”的理解

当前子任务的主位是验证新模块，而不是先重构新模块再验证。因此第一版实现应尽量不修改 `fealpy/mesh` 源码。

但当前仓库已有新接口风格的 box 区域网格生成入口 `fealpy/mesher/box.py`，其中已经实现了三维 box 区域的 `Box3d`。因此当前算例可以在 `fealpy/mesher/box.py` 中参考 `Box3d` 增加一个二维 box 区域三角形网格类，例如 `Box2d`，用于提供当前算例的实验网格来源。

这类修改属于 `fealpy/mesher` 的实验载体补充，不属于对 `fealpy/mesh` 核心数据结构、schema、storage、topology 或 view 的重构。后续报告中应明确说明：二维 box 三角网格生成能力是在本子任务中为验证算例补充的 mesher 能力，而不是当前新 `fealpy/mesh` 模块本体原先已经具备的能力。

但是，如果在实现过程中发现当前模块存在明显 bug，例如二维三角形几何量计算直接失败、关系查询接口无法返回已有关系、边界标记与关系不一致等，应在验证记录中如实记录，并作为后续接口完善任务的输入。是否修复这些问题，不应由当前算法设计文档直接决定。

---

## 三、总体方案

### 3.1 可选路线

#### 方案 A：新模块主流程 + 算例内最小补充

主流程使用新 `fealpy/mesh` 模块构造或承载网格，并从新模块中获取节点、三角形和已存在的实体关系。新模块暂未提供或未稳定暴露的能力，例如 `edge -> tri`、`cell -> cell`、二维有向面积、路径记录等，在算例或测试程序中实现。

优点：

- 不把当前子任务变成新模块开发任务。
- 能真实暴露当前模块已经具备和暂时缺失的能力。
- 方便在报告中区分模块能力与算例辅助能力。

缺点：

- 算例文件会包含一些后续可复用的辅助逻辑。
- 如果辅助逻辑过多，容易削弱“新模块已经能用”的说服力。

#### 方案 B：先补齐新模块 API，再写 Line Walk 算例

直接在 `fealpy/mesh` 中新增 `edge_to_cell`、`cell_to_cell`、二维三角形面积、点定位辅助接口等能力，再编写 Line Walk 算例。

优点：

- 算例代码更简洁。
- 对后续算法迁移有直接收益。

缺点：

- 容易越出当前子任务边界。
- 会把“验证当前模块”变成“先修改模块再验证”。
- 新增 API 本身也需要额外测试和接口讨论。

#### 方案 C：使用旧网格模块或 legacy 接口辅助

通过旧版 FEALPy 网格接口获取邻接、边界、几何量等能力，然后只在表面上使用新模块作为输入。

优点：

- 最快得到一个可运行的点定位程序。

缺点：

- 违背当前子任务“主流程走新模块”的要求。
- 无法判断新模块本身是否足以支撑典型算法。
- 会混淆验证结论。

### 3.2 推荐方案

推荐采用 **方案 A：新模块主流程 + 算例内最小补充**。

后续实现时应坚持两条规则：

1. 凡是新模块已经直接提供的能力，必须从新模块入口取得。
2. 凡是新模块没有提供或尚未稳定暴露的能力，只能在算例或测试程序中做最小补充，并在文档和报告中显式列出。

---

## 四、新模块参与方式与接口适配层

### 4.1 概念命名说明

架构文档中强调新 Mesh 系统分为三类职责：

- 规则模板层：定义实体规则和形状操作。
- 拓扑存储层：保存节点坐标、实体连接关系和实体关系。
- 计算视图层：向用户提供统一访问入口。

在具体代码中，实体数据块可能使用 `EntitySector`、`EntityBlock` 或其它实现命名。本文后续统一使用“实体块 / sector”表达这一概念；涉及具体代码时，以仓库当前实现的实际类名为准。

### 4.2 新模块应直接参与的对象

本算例应尽量通过新模块获得以下数据或能力：

| 能力 | 用途 | 是否必须来自新模块 |
|---|---|---|
| 节点坐标 `positions` | 计算有向面积和重心坐标 | 是 |
| 三角形连接关系 `tri` | 获得当前 cell 的三个顶点 | 是 |
| 三角形实体视图 | 访问三角形数量、连接关系和关系查询 | 是 |
| 边实体视图 | 获取边数量，辅助构造邻接 | 优先来自新模块 |
| `tri -> edge` 关系 | 将三角形局部边映射到全局边 | 优先来自新模块 |
| `edge -> node` 关系 | 验证全局边对应的两个节点 | 优先来自新模块 |
| 边界信息 | 判断穿出边界 | 若新模块提供则使用，否则由邻接计数派生 |

### 4.3 接口适配层设计

为了避免算法主流程被当前仓库接口细节绑死，建议实现一个很薄的适配函数：

```python
def extract_triangle_mesh_data(mesh):
    """
    从新 Mesh 对象中提取 Line Walk 所需的最小数据。

    Returns
    -------
    data : TriangleMeshData
        包含 positions、tri、tri_to_edge、edge_to_node 等字段。
    """
```

建议定义：

```python
@dataclass
class TriangleMeshData:
    positions: Tensor          # shape = (Nnode, 2)
    tri: Tensor                # shape = (Ncell, 3)
    tri_to_edge: Tensor | None # shape = (Ncell, 3)
    edge_to_node: Tensor | None
    boundary_edge_mask: Tensor | None
```

该适配层的职责是：

1. 把新 Mesh 当前的访问路径集中封装起来。
2. 明确哪些数据是从新模块中拿到的。
3. 允许后续仓库接口变化时只修改适配层，而不修改 Line Walk 主算法。

该适配层不应：

1. 调用旧网格模块完成核心拓扑查询。
2. 把新 Mesh 转换成旧 Mesh 后再执行算法。
3. 在内部完成 Line Walk 主循环。

---

## 五、程序资产建议

当前设计建议后续形成两个代码资产和两个文档资产。

### 5.1 算例程序

建议路径：

```text
kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/examples/triangle_line_walk_on_new_mesh.py
```

职责：

- 调用 `fealpy/mesher/box.py` 中新增的二维 box 三角网格生成类。
- 使用新模块承载该网格并构造必要拓扑。
- 从新模块入口提取节点、三角形和关系数据。
- 执行若干代表点的 Line Walk。
- 打印定位结果与路径记录。

该文件定位为“可读、可运行、可复现”的算例入口。

### 5.2 最小测试

建议路径：

```text
kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/tests/test_triangle_line_walk_on_new_mesh.py
```

职责：

- 用断言验证网格实体数量。
- 验证 `tri -> edge`、边界边数量和邻接关系。
- 验证内部点、共享边附近点、边界点、网格外点的定位结果。
- 验证路径中每一步穿越边与符号判断一致。

### 5.3 验证说明

建议路径：

```text
kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/triangle_line_walk_case_on_new_mesh_validation_record.md
```

职责：

- 记录运行命令。
- 记录测试网格规模。
- 记录测试点和预期结果。
- 记录运行输出摘要。
- 说明当前验证覆盖与未覆盖范围。
- 说明新模块直接提供能力与算例补充能力。

### 5.4 阶段性收口说明

建议路径：

```text
kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/triangle_line_walk_case_on_new_mesh_summary.md
```

职责：

- 汇总架构审查、算法实现、验证记录。
- 回答子任务完成判据中的阶段性问题。
- 列出后续建议沉入新网格模块的能力。

---

## 六、实验网格设计

### 6.1 默认网格

默认使用 box 区域 `[0, 1] x [0, 1]` 上的均匀三角形网格：

- `nx = 10`
- `ny = 10`
- 节点数量：`(nx + 1) * (ny + 1) = 121`
- 三角形数量：`2 * nx * ny = 200`
- 每个矩形单元沿同一方向的对角线剖分为两个三角形。

每个矩形局部节点记为：

```text
n01 ---- n11
 |  \     |
 |     \  |
n00 ---- n10
```

建议三角形连接方式：

```python
[n00, n10, n01]
[n10, n11, n01]
```

在标准 `x` 向右、`y` 向上的二维坐标中，这两个三角形均为逆时针方向。

### 6.2 二维 box 三角网格生成类

现有 `fealpy/mesher/box.py` 中已有新接口风格的三维 box 网格类 `Box3d`，其基本路径是：

1. 在 `initialize()` 中生成节点坐标和高维实体连接关系；
2. 在 `tetrahedralize()` 或 `prismatize()` 中把实体连接关系转换为对应单元；
3. 使用 `MeshBlock`、`EntitySector` 和 `TopologyBuilder.construct()` 构造新 `Mesh` 对象。

当前子任务建议参考该实现，在同一文件中补充二维 box 区域的三角形网格类，例如：

```python
class Box2d:
    def __init__(
        self,
        box: list[float] = [0, 1, 0, 1],
        nx: int = 10,
        ny: int = 10,
    ) -> None:
        ...

    def initialize(self):
        ...

    def triangulate(self):
        ...
```

职责：

1. `initialize()` 生成二维节点坐标 `node` 和规则四边形连接关系 `cell`。
2. `triangulate()` 将每个四边形沿固定对角线剖分成两个三角形。
3. `triangulate()` 使用新 `fealpy/mesh` 模块承载 `node` 和三角形连接关系。
4. `triangulate()` 调用 `TopologyBuilder.construct(storage)` 生成低维实体和关系。
5. `triangulate()` 返回新模块的 `Mesh` 对象。

推荐三角形剖分方式仍采用第 6.1 节中的：

```python
[n00, n10, n01]
[n10, n11, n01]
```

该类属于当前子任务中补充的 `fealpy/mesher` 实验载体能力。它可以被当前目录下的 examples 和 tests 调用，但不应在报告中声称为完整通用 mesher。

### 6.3 网格构造结果记录

验证记录中应至少记录：

```markdown
- 网格区域：`[0, 1] x [0, 1]`
- 剖分参数：`nx = 10, ny = 10`
- 节点数量：121
- 三角形数量：200
- 网格生成入口：`fealpy.mesher.box.Box2d.triangulate()`
- 是否使用新 Mesh 保存 `positions`：是 / 否
- 是否使用新 Mesh 保存三角形实体：是 / 否
- 是否由新 Mesh 提供 `tri -> edge`：是 / 否
- 是否由新 Mesh 提供边界信息：是 / 否
- 是否由算例临时派生 `edge -> tri`：是 / 否
- 是否由算例临时派生 `cell -> cell`：是 / 否
```

---

## 七、邻接派生设计

Line Walk 需要从当前三角形穿过某条边进入相邻三角形。因此至少需要 `cell_neighbors`：

```python
neighbors[c, le] = 与三角形 c 的局部边 le 相邻的另一个三角形编号
```

若该局部边是边界边，则：

```python
neighbors[c, le] = -1
```

### 7.1 输入数据

理想情况下，从新模块获得：

```python
tri_to_edge: Tensor  # shape = (Ncell, 3)
edge_to_node: Tensor # shape = (Nedge, 2)
```

如果当前新模块没有直接提供 `tri_to_edge`，则需要在验证记录中明确说明：当前算例无法直接验证基于新模块关系系统的穿边步进能力，只能退化为从三角形连接关系临时构造边关系。该情况应作为重要接口缺口记录。

### 7.2 构造 `edge_to_cells`

算例中建议实现：

```python
def build_edge_to_cells(tri_to_edge: Tensor, nedge: int) -> list[list[tuple[int, int]]]:
    ...
```

返回值：

- 长度为 `nedge` 的 Python list。
- 每个元素记录共享该 edge 的三角形及其局部边编号。
- 每个条目形如 `(cell_index, local_edge_index)`。

示例：

```python
edge_to_cells[e] = [(c0, le0), (c1, le1)]
```

表示全局边 `e` 被两个三角形共享，分别位于 `c0` 的局部边 `le0` 和 `c1` 的局部边 `le1`。

如果某条边关联的三角形数量超过 2，说明当前网格存在非流形边。第一版 Line Walk 不处理非流形边，应直接返回错误或在测试中断言失败。

### 7.3 构造 `cell_neighbors`

算例中建议实现：

```python
def build_cell_neighbors(
    tri_to_edge: Tensor,
    edge_to_cells: list[list[tuple[int, int]]],
) -> Tensor:
    ...
```

返回：

- 形状为 `(Ncell, 3)` 的整数数组。
- `neighbors[c, le]` 表示从三角形 `c` 穿过局部边 `le` 后到达的相邻三角形。
- 若该边是边界边，则值为 `-1`。

该邻接表只用于当前 Line Walk 算例。后续若沉入模块，可考虑扩展为通用 `cell_to_cell()` 查询，或作为 `EntityRelation` 的反向推断能力。

---

## 八、几何判断设计

### 8.1 不直接依赖二维三角形 `measure()`

当前 Line Walk 算法只需要判断点相对于三角形各边的位置，不需要依赖 `TriangleSchema.measure()`。

如果架构审查或代码实验已经发现当前二维三角形 `measure()` 存在问题，应在验证记录中单独列出，但 Line Walk 第一版不应把修复 `measure()` 作为前置条件。

### 8.2 二维有向面积函数

算例中实现二维有向面积函数：

```python
def orient2d(a: Tensor, b: Tensor, p: Tensor) -> Tensor:
    return (b[..., 0] - a[..., 0]) * (p[..., 1] - a[..., 1]) - \
           (b[..., 1] - a[..., 1]) * (p[..., 0] - a[..., 0])
```

其中，`orient2d(a, b, p)` 表示从 `a -> b` 到 `a -> p` 的二维叉积。

### 8.3 使用局部重心坐标符号判断

对当前三角形：

```python
tri[c] = [v0, v1, v2]
```

定义三条“对边”：

| 重心坐标 | 对应局部边 | 对边顶点 |
|---|---|---|
| `lambda0` | `(v1, v2)` | `v0` |
| `lambda1` | `(v2, v0)` | `v1` |
| `lambda2` | `(v0, v1)` | `v2` |

计算方式为：

```python
lambda0 = orient2d(p1, p2, p) / orient2d(p1, p2, p0)
lambda1 = orient2d(p2, p0, p) / orient2d(p2, p0, p1)
lambda2 = orient2d(p0, p1, p) / orient2d(p0, p1, p2)
```

其中：

```python
p0 = positions[v0]
p1 = positions[v1]
p2 = positions[v2]
```

这种写法的优点是：

1. 判断规则不依赖三角形整体是顺时针还是逆时针。
2. 每个 `lambda_i` 都与一个明确的对边对应。
3. 若 `lambda_i < -tol`，说明点位于局部边 `i` 的外侧，应尝试穿过该边。

### 8.4 局部边到新模块局部边编号的映射

Line Walk 使用的三条局部边是：

```python
walk_edges = [
    (1, 2),  # opposite vertex 0
    (2, 0),  # opposite vertex 1
    (0, 1),  # opposite vertex 2
]
```

但是新模块中 `TriangleSchema.local_faces["edge"]` 的顺序不一定与上述顺序一致。因此不能隐式假设：

```python
neighbors[c, i]
```

中的 `i` 就等于 Line Walk 的 `lambda_i`。

应显式构造映射：

```python
def build_walk_edge_to_schema_local_edge(schema_edges: list[list[int]]) -> list[int]:
    """
    将 Line Walk 使用的对边顺序映射到 schema 中的局部边编号。
    """
    walk_edges = [(1, 2), (2, 0), (0, 1)]
    mapping = []
    for e in walk_edges:
        eset = set(e)
        for k, se in enumerate(schema_edges):
            if set(se) == eset:
                mapping.append(k)
                break
        else:
            raise ValueError(f"walk edge {e} not found in schema edges")
    return mapping
```

若当前 `TriangleSchema.local_faces["edge"] = [[0, 1], [0, 2], [1, 2]]`，则映射结果应为：

```python
walk_to_schema_le = [2, 1, 0]
```

含义是：

- `lambda0` 对应局部边 `(1, 2)`，在 schema 中是第 2 条边；
- `lambda1` 对应局部边 `(2, 0)`，在 schema 中是第 1 条边；
- `lambda2` 对应局部边 `(0, 1)`，在 schema 中是第 0 条边。

> 注意：这一步是当前算法设计的关键点。只要 schema 的局部边顺序与算法边顺序不一致，就必须显式映射，否则 Line Walk 很容易沿错误邻接边步进。

---

## 九、Line Walk 主算法设计

### 9.1 结果对象

建议定义轻量结果对象：

```python
@dataclass
class LineWalkStep:
    cell: int
    lambdas: list[float]
    walk_edge_index: int | None
    schema_local_edge: int | None
    global_edge: int | None
    next_cell: int | None


@dataclass
class LineWalkResult:
    point: Tensor
    start_cell: int
    located_cell: int | None
    status: str
    lambdas: list[float] | None
    path: list[int]
    steps: list[LineWalkStep]
    message: str
```

状态建议使用：

| 状态 | 含义 |
|---|---|
| `inside` | 点在某个三角形内部 |
| `on_edge` | 点落在当前三角形某条边上 |
| `on_vertex` | 点落在当前三角形某个顶点附近 |
| `outside` | 点从边界边穿出，判断为网格外 |
| `cycle` | 步进过程中出现重复 cell |
| `max_steps` | 超过最大步数仍未定位 |
| `degenerate_cell` | 遇到退化三角形 |
| `invalid_input` | 输入 cell 或点非法 |

### 9.2 函数签名

建议函数签名：

```python
def line_walk_locate(
    mesh: Mesh,
    point: Tensor,
    *,
    start_cell: int = 0,
    tol: float = 1.0e-12,
    max_steps: int | None = None,
) -> LineWalkResult:
    ...
```

输入语义：

- `mesh`：由新模块构造或承载的二维三角形网格。
- `point`：待定位二维点。
- `start_cell`：起始三角形编号。
- `tol`：几何判断容差。
- `max_steps`：最大步数；默认可设为 `Ncell + 1` 或 `2 * Ncell`。

### 9.3 每一步计算

对当前 cell `c`：

1. 读取三角形顶点：

```python
cell = tri[c]
verts = positions[cell]
```

2. 计算三个局部重心坐标：

```python
lambdas = triangle_lambdas(verts, point)
```

3. 判断点是否在当前三角形内：

```python
if min(lambdas) >= -tol:
    # inside / on_edge / on_vertex
```

4. 若存在负值，选择最负的 `lambda_i` 对应的对边作为穿越边：

```python
walk_edge_index = argmin(lambdas)
schema_local_edge = walk_to_schema_le[walk_edge_index]
global_edge = tri_to_edge[c, schema_local_edge]
next_cell = neighbors[c, schema_local_edge]
```

5. 若 `next_cell == -1`：

- 说明目标点沿该方向穿出网格边界；
- 返回 `status = "outside"`。

6. 否则进入 `next_cell`，继续循环。

### 9.4 终止条件

算法终止条件包括：

1. 找到包含点的 cell：返回 `inside`、`on_edge` 或 `on_vertex`。
2. 穿越边界离开网格：返回 `outside`。
3. 达到最大步数：返回 `max_steps`。
4. 检测到重复 cell 且仍未定位：返回 `cycle`。
5. 遇到退化三角形：返回 `degenerate_cell`。

### 9.5 主流程伪代码

```python
def line_walk_locate(mesh, point, start_cell=0, tol=1.0e-12, max_steps=None):
    data = extract_triangle_mesh_data(mesh)

    positions = data.positions
    tri = data.tri
    tri_to_edge = data.tri_to_edge

    if tri_to_edge is None:
        raise RuntimeError("Line Walk requires tri_to_edge relation or an explicit fallback builder.")

    edge_to_cells = build_edge_to_cells(tri_to_edge, nedge=...)
    neighbors = build_cell_neighbors(tri_to_edge, edge_to_cells)

    schema_edges = get_triangle_schema_edges(mesh)
    walk_to_schema_le = build_walk_edge_to_schema_local_edge(schema_edges)

    ncell = tri.shape[0]
    if max_steps is None:
        max_steps = ncell + 1

    current = start_cell
    visited = set()
    path = []
    steps = []

    for _ in range(max_steps):
        if current < 0 or current >= ncell:
            return invalid_input_result(...)

        if current in visited:
            return cycle_result(...)

        visited.add(current)
        path.append(current)

        verts = positions[tri[current]]
        lambdas = triangle_lambdas(verts, point)

        if is_degenerate(verts):
            return degenerate_cell_result(...)

        if min(lambdas) >= -tol:
            status = classify_inside_boundary(lambdas, tol)
            return success_result(status, current, lambdas, path, steps)

        walk_edge_index = argmin(lambdas)
        schema_le = walk_to_schema_le[walk_edge_index]
        global_edge = tri_to_edge[current, schema_le]
        next_cell = neighbors[current, schema_le]

        steps.append(LineWalkStep(
            cell=current,
            lambdas=lambdas,
            walk_edge_index=walk_edge_index,
            schema_local_edge=schema_le,
            global_edge=global_edge,
            next_cell=next_cell,
        ))

        if next_cell < 0:
            return outside_result(path, steps, lambdas)

        current = next_cell

    return max_steps_result(path, steps)
```

---

## 十、正确性验证设计

### 10.1 网格结构验证

对 `nx = ny = 10` 的默认网格，建议验证：

- 节点数量为 121。
- 三角形数量为 200。
- `tri_to_edge` 的形状为 `(200, 3)`。
- 每个三角形恰有 3 条局部边。
- 边界边数量为 `2 * nx + 2 * ny = 40`。
- 内部边有两个相邻三角形。
- 边界边只有一个相邻三角形。
- `neighbors` 的形状为 `(200, 3)`。
- 对非边界邻接 `neighbors[c, le] = nb`，`c` 和 `nb` 应共享两个节点。

边总数可以作为运行记录项，不必在第一版中作为强断言；如果要断言，应先明确当前剖分方式对应的边数公式，并与实际构造一致。

### 10.2 点定位验证

建议测试点分组：

| 类型 | 示例点 | 预期 |
|---|---|---|
| 内部点 | `(0.23, 0.37)` | 定位到某个包含该点的 cell |
| 单元重心点 | 取若干 cell 的 barycenter | 必须定位到包含该重心的 cell |
| 共享边附近点 | `(0.5, 0.5)` 或加微小偏移 | 定位到合法相邻 cell 之一 |
| 边界边上点 | `(0.0, 0.5)` | `on_edge` 或返回包含该点的边界 cell |
| 顶点点 | `(1.0, 1.0)` | `on_vertex` 或返回包含该点的边界 cell |
| 网格外点 | `(-0.1, 0.5)` | `outside` |
| 网格外点 | `(0.5, 1.1)` | `outside` |

对于共享边或顶点上的点，不建议第一版要求唯一 cell 编号，因为浮点容差和边界归属约定会影响结果。第一版应要求：

- 返回状态明确。
- 返回 cell 的几何包含关系在容差内成立。
- 路径不死循环。

### 10.3 Brute-force Oracle

为了验证 Line Walk 结果，建议实现一个只用于测试的 brute-force oracle：

```python
def brute_force_locate(positions, tri, point, tol=1.0e-12) -> list[int]:
    """
    遍历所有三角形，返回所有包含 point 的候选 cell。
    """
```

验证规则：

1. 若 Line Walk 返回 `inside`、`on_edge` 或 `on_vertex`：
   - `located_cell` 必须属于 `oracle_cells`。
2. 若 Line Walk 返回 `outside`：
   - `oracle_cells` 必须为空。
3. 若 Line Walk 返回 `cycle` 或 `max_steps`：
   - 当前测试不通过，并记录失败路径。
4. 若 oracle 返回多个 cell：
   - 说明点可能在共享边或共享顶点上；
   - Line Walk 返回其中任意一个候选 cell 均可接受。

### 10.4 路径一致性验证

对每个发生步进的路径条目，验证：

1. `walk_edge_index == argmin(lambdas)`。
2. `lambdas[walk_edge_index] < -tol`。
3. `schema_local_edge == walk_to_schema_le[walk_edge_index]`。
4. `global_edge == tri_to_edge[cell, schema_local_edge]`。
5. 若 `next_cell >= 0`，则 `cell` 与 `next_cell` 共享 `global_edge` 对应的两个节点。
6. 若 `next_cell == -1`，则该边应为边界边，或在 `edge_to_cells` 中只有一个相邻 cell。

### 10.5 新模块参与度验证

最终验证说明中必须列出：

新模块直接提供的能力，例如：

- `positions` 数据。
- 三角形实体数据。
- 边实体数据。
- `tri -> edge` 关系。
- `edge -> node` 关系。
- 边界信息。
- `Mesh / EntityView` 查询入口。

算例或测试程序补充的能力，例如：

- `edge_to_cells` 派生。
- `cell_neighbors` 派生。
- `orient2d`。
- 局部重心坐标符号判断。
- Line Walk 主循环。
- 路径记录与结果对象。
- Brute-force oracle。

本子任务在 `fealpy/mesher` 中补充的实验载体能力，例如：

- `Box2d.initialize()`。
- `Box2d.triangulate()`。

如果某项“新模块直接提供能力”在当前仓库中并不存在，应从该列表中删除，并转入“算例补充能力”或“后续模块缺口”。

---

## 十一、后续可沉入新网格模块的能力

当前测试程序中实现的能力，后续可按优先级考虑沉入新网格模块。

### 11.1 高优先级

- `cell_to_cell` 邻接查询。
- relation 反向推断，例如从 `tri -> edge` 推断 `edge -> tri`。
- 二维三角形几何量计算，例如面积、重心、法向或有向面积。
- `TriangleSchema` 的局部边语义说明。
- opposite-vertex 辅助表，即从局部顶点编号得到其对边编号。

### 11.2 中优先级

- 通用 `entity_to_entity` 邻接查询。
- 非流形边检测与报告，例如一条边邻接超过两个 cell。
- 边界信息缓存与拓扑修改后的失效机制。
- `EntityView` 对未实现几何接口提供更明确错误信息。

### 11.3 低优先级

- 更通用的 box 区域二维 mesher 参数化能力，例如不同对角线方向、返回裸连接关系或多种单元类型。
- 通用点定位接口。
- 多起点 Line Walk 或空间索引加速。
- 退化点、边上点、顶点点的统一归属策略。

---

## 十二、实施任务拆分

### Task 1：补充二维 box 三角网格生成类

文件：

```text
fealpy/mesher/box.py
```

步骤：

1. 参考已有 `Box3d` 增加 `Box2d`。
2. 实现 `Box2d.initialize()`，生成二维节点坐标和规则四边形连接关系。
3. 实现 `Box2d.triangulate()`，把每个四边形剖分为两个三角形。
4. 在 `triangulate()` 中使用 `MeshBlock`、`EntitySector("tri", tri)` 和 `TopologyBuilder.construct()` 返回新 `Mesh` 对象。
5. 确认 `Box2d(nx=10, ny=10).triangulate()` 可以得到 121 个节点和 200 个三角形。

验收：

- 二维 box 三角网格可以通过 `fealpy/mesher/box.py` 的新类生成。
- 生成结果直接使用新 `fealpy/mesh` 模块承载和构造拓扑。

### Task 2：建立当前子任务算例入口

文件：

```text
kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/examples/triangle_line_walk_on_new_mesh.py
```

步骤：

1. 从 `fealpy.mesher.box` 导入 `Box2d`。
2. 调用 `Box2d(nx=10, ny=10).triangulate()` 获得新 `Mesh` 对象。
3. 打印或断言节点、三角形、边数量。
4. 确认 `tri -> edge` relation 是否可查询。

验收：

- 当前子任务目录下的算例可以运行。
- 输出能证明新模块参与了网格事实存储和拓扑构造。

### Task 3：实现接口适配层

文件：

```text
kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/examples/triangle_line_walk_on_new_mesh.py
```

步骤：

1. 实现 `TriangleMeshData`。
2. 实现 `extract_triangle_mesh_data(mesh)`。
3. 将新模块具体 API 调用集中封装在该函数中。
4. 在运行输出中列出成功提取的数据项。

验收：

- 后续 Line Walk 主算法不直接依赖散落的仓库接口调用。
- 哪些数据来自新模块清楚可查。

### Task 4：实现邻接派生辅助函数

文件：

```text
kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/examples/triangle_line_walk_on_new_mesh.py
```

步骤：

1. 实现 `build_edge_to_cells()`。
2. 实现 `build_cell_neighbors()`。
3. 验证内部边有两个相邻 cell，边界边只有一个相邻 cell。
4. 若新模块提供边界信息，则与邻接计数结果做一致性检查。

验收：

- 每个 cell 有 3 个邻接槽位。
- 边界槽位为 `-1`。
- 内部共享边能找到对侧 cell。

### Task 5：实现二维符号判断与局部边映射

文件：

```text
kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/examples/triangle_line_walk_on_new_mesh.py
```

步骤：

1. 实现 `orient2d()`。
2. 实现 `triangle_lambdas()`。
3. 定义 `walk_edges = [(1, 2), (2, 0), (0, 1)]`。
4. 从 `TriangleSchema.local_faces["edge"]` 或等价接口派生 `walk_to_schema_le`。
5. 对若干 cell 重心验证三个 `lambda` 均非负。

验收：

- 二维符号判断不依赖三角形整体朝向。
- 局部边映射显式可检查。

### Task 6：实现 Line Walk 主循环

文件：

```text
kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/examples/triangle_line_walk_on_new_mesh.py
```

步骤：

1. 定义 `LineWalkStep` 和 `LineWalkResult`。
2. 实现 `line_walk_locate()`。
3. 记录路径、重心坐标、穿越边和状态。
4. 对内部点和外部点进行手动运行检查。

验收：

- 内部点能返回合法 cell。
- 外部点能返回 `outside`。
- 路径长度有限，不出现无限循环。

### Task 7：补充最小断言测试

文件：

```text
kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/tests/test_triangle_line_walk_on_new_mesh.py
```

步骤：

1. 测试默认网格实体数量和边界边数量。
2. 测试 `edge_to_cells` 与 `cell_neighbors`。
3. 测试若干 cell 重心点定位。
4. 测试网格外点定位。
5. 测试路径步进与符号选择一致。
6. 使用 brute-force oracle 校验 Line Walk 输出。

验收：

```bash
python -m pytest kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/tests/test_triangle_line_walk_on_new_mesh.py
```

能够复现通过。

### Task 8：形成验证记录与阶段摘要

文件：

```text
triangle_line_walk_case_on_new_mesh_validation_record.md
triangle_line_walk_case_on_new_mesh_summary.md
```

步骤：

1. 记录运行命令与输出摘要。
2. 记录当前验证覆盖项。
3. 记录未覆盖项。
4. 汇总哪些能力由新模块提供，哪些能力由测试程序补充。
5. 汇总后续建议沉入新网格模块的能力。

验收：

- 文档能支撑当前子任务对象级完成判据。

---

## 十三、风险与规避

### R1：点在边上或顶点上时 cell 归属不唯一

规避：

- 第一版测试中不强制唯一 cell。
- 对边上点和顶点点，只验证返回 cell 合法且符号在容差内。

### R2：局部边顺序误用导致步进方向错误

规避：

- 必须通过无向集合匹配生成 `walk_to_schema_le`。
- 路径一致性测试检查 `lambda` 最负边与实际穿越边一致。

### R3：三角形朝向不一致导致符号判断错误

规避：

- 使用带分母归一化的局部重心坐标判断，而不是只假设所有三角形均为逆时针。
- 对默认构造网格额外检查三角形面积符号是否一致。

### R4：二维几何接口当前不可直接用

规避：

- 当前算例使用局部 `orient2d()` 与 `triangle_lambdas()`。
- 最终报告明确说明这属于算例补充能力。
- 若确认新模块二维 `measure()` 存在问题，将其列为后续修复候选。

### R5：边界缓存或关系缓存可能过期

规避：

- 算例约定先完成拓扑构造，再查询边界与关系。
- 不在同一个 Mesh 对象上动态修改拓扑后复用旧关系。

### R6：辅助函数过多导致验证边界扩散

规避：

- 辅助函数只服务当前三角形 Line Walk。
- 不在当前任务中抽象为通用库。
- 在报告中把它们列为“算例补充能力”。

---

## 十四、完成判据

当以下条件满足时，可认为当前算法设计进入可实施状态：

1. 算例主流程明确使用新 `fealpy/mesh` 模块承载和访问网格。
2. 当前模块缺少的能力已有算例内最小补充方案。
3. Line Walk 的几何符号、局部边映射、邻接步进和终止条件已经明确。
4. 验证点组合、brute-force oracle 和路径一致性检查已经明确。
5. 新模块直接提供能力与算例补充能力能够在验证记录中分开说明。
6. 后续可沉入新网格模块的能力已经单独列出。
7. 当前设计不会把旧网格模块作为主流程依赖。

---

## 十五、阶段性结论

本设计采用“新模块主流程 + 算例内最小补充”的路线。该路线能在不大规模修改新网格模块源代码的前提下，验证当前新模块是否足以支撑一个真实的三角形网格算法算例。

本算例预计会直接验证新模块的实体存储、拓扑构造、关系查询和视图访问能力；同时会暴露当前模块在反向邻接、`cell_to_cell` 查询、二维三角形几何量、局部边语义和边界表达方面的接口缺口。

这些缺口不一定阻塞当前子任务，但应作为后续接口完善、算法迁移和测试系统建设的候选输入。当前算例的阶段性价值不在于证明新 Mesh 模块已经完整成熟，而在于给出一个小而完整、可运行、可解释、可复现的先验证据。
