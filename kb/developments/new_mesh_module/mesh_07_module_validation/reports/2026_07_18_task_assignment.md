# Mesh 07 开发任务分配

> 依据：`reports/known_problems.md`、`reports/2026_07_18.md`、`validations/fvm_validation_record.md`。
>
> 本文件只分配已形成明确决策、且可以在一天内完成的开发任务；未决策项不进入本轮任务。问题编号用于追踪验证反馈，不直接等同于人员任务。

## 一、分配原则

- 本轮安排 3 名开发人员，任务之间按代码边界拆分，可并行开展。
- 每项任务都必须包含代码修改、针对性测试和实际 pytest 结果。
- 不允许由上层 FVM 重复实现面法向或积分权重；几何算法和积分算法仍由 Mesh 模块负责。
- 不改变 `barycenter()` 的语义，不处理尚未决策的旧接口迁移、NodeMesh 特殊算法、from_<...> 构造器补全等事项。
- 同一代码文件原则上只由一个任务负责人修改，避免并行冲突。

## 二、任务总览

| 任务    | 负责人    | 主题                     | 主要代码范围                                         | 并行关系                          |
| ----- | ------ | ---------------------- | ---------------------------------------------- | ----------------------------- |
| M07-A | 开发人员 A | 四边形 OFace 顺序与几何算法      | `quadrilateral.py`、`mesher/box.py`             | 可独立开始                         |
| M07-C | 开发人员 C | Jacobian 加权积分、契约及其单元测试 | `classic/base.py`、`entity_schema.py`、积分测试      | 可独立开始；不得修改 A 的四边形算法           |
| M07-D | 开发人员 D | 恢复六面体原 FEALPy 顶点排序     | `hexahedron.py`、必要的 `mesher/box.py`/拓扑关系及六面体测试 | 可独立开始；与 A 共享 `box.py` 时须先协调边界 |

A、C、D 各自负责实现和对应单元测试。测试必须归属于被测 Mesh 模块，文件名和测试函数名忠实描述模块、接口或数学性质，不得使用任务编号、验证活动名称或 FVM 等具体领域名称。三项任务完成后再运行相关测试集合进行交叉核验。

---

## 三、M07-A：四边形 OFace 顺序与几何算法

### 目标

以 `OFace` 的环状节点顺序作为四边形实体的唯一几何节点顺序，修复四边形映射、测度、法向、切向等算法中隐式使用 `[0, 1, 3, 2]` 重排的问题，并同步保证 Box2d 生成的 quad 节点按该约定排列。

### 修改范围

- `fealpy/mesh/schema/classic/quadrilateral.py`
- `fealpy/mesher/box.py`
- `tests/mesh/unit/schema/test_quadrilateral_schema.py`
- `tests/mesher/unit/test_box.py`

不得修改：`fealpy/mesh/view/fealpy_api.py`、`fealpy/mesh/schema/classic/base.py`、其它负责人负责的实现文件。

### 实现要求

1. 先确认并记录 `QuadrilateralSchema.OFace["segment"]` 对应的环状顺序：四个顶点连续连接，且采用项目已决策的逆时针约定。
2. 统一检查并修正以下函数的节点解释：
   - `bc_to_point()`
   - `grad_shape_function_*()`
   - `grad_lambda()`
   - `jacobi_matrix()`
   - `measure()`
   - `normal()`
   - `tangent()`
   - `barycenter()`
3. 不用一个固定排列掩盖 `OFace` 与几何算法的冲突；所有函数必须使用同一局部节点语义。
4. 三维平面四边形的法向应非零，并与环状顺序一致；二维四边形的已有测度、切向和映射行为不能回归。
5. 检查 `Box2d.initialize()` / `quadrangulate()` 的 cell 排列，使生成的 quad 符合相同的环状顺序；若当前已经符合，只补充测试并不要无谓改动。

### 必须测试

至少增加或更新以下断言：

- 直接构造的规则矩形 quad：映射、面积、切向和法向正确。
- 验证记录中的非仿射梯形：面积为 `3/2`，节点顺序为环状顺序。
- `Box2d(nx=1, ny=1).quadrangulate()` 生成的 cell 顺序符合 `OFace` 约定。
- 六面体生成的真实 quad face 不因节点重排而出现零法向；这里只验证 `Entity("face").normal()` 的几何值。

### 验证命令

```bash
pytest -q tests/mesh/unit/schema/test_quadrilateral_schema.py tests/mesher/unit/test_box.py
```

完成反馈必须包含：修改文件、节点顺序依据、测试命令和真实通过数量。

---

## 四、M07-C：Jacobian 加权积分及统一契约

### 目标

落实已决策的数学方案：公共 `integral()` 使用每个实体、每个积分点的 Jacobian 测度因子，而不是使用单个常数 `measure()` 代替；同时明确 `jacobi_matrix()` 的形状契约。

### 修改范围

- `fealpy/mesh/schema/classic/base.py`
- `fealpy/mesh/schema/entity_schema.py`
- 必要时只修改与 Jacobian 返回形状直接相关的 concrete schema；如无需修改，不要扩大范围。
- 新增或修改模块测试文件：`tests/mesh/unit/test_entity_integral.py`

- 不得修改 A 负责的四边形法向实现和测试文件。`fealpy/mesh/schema/entity_schema.py` 由 M07-C 独占修改，负责同时明确 `normal()` 和 `jacobi_matrix()` 的公共契约。

### 数学与实现要求

1. 明确参考域到物理域的积分公式：
   `integral_K(f) = sum_q w_q * f(F_K(xi_q)) * J_measure(K, xi_q)`。
2. 根据 Jacobian 的形状 `(NE, NQ, GD, ref_dim)` 计算测度因子：
   - `GD == ref_dim` 的体映射使用 `abs(det(J))`；
   - 嵌入曲面使用 Gram 矩阵 `sqrt(det(J^T J))`；
   - 线实体使用切向 Jacobian 的范数。
3. 优先使用 backend 支持的张量运算、`einsum` 和批量线性代数，不为实体或积分点写 Python 循环。
4. `index` 必须同时作用于映射、Jacobian 和积分结果，不能只切 `measure`。
5. 保证常数函数积分等于 `measure()`；对仿射实体，逐点 Jacobian 方案应与旧结果一致。
6. 不改变函数参数语义，不修改 `barycenter()`。

### 必须测试

- 验证记录中的非仿射梯形：
  - `measure() == 3/2`；
  - `integral(x) == 7/6`；
  - `integral(y) == 2/3`。
- 单位或规则四边形：常数函数积分等于面积，仿射积分保持原结果。
- 三角形、四面体至少各有一个常数函数积分回归测试。
- `index=None` 与子集 index 的积分结果和形状正确。
- 测试中用显式 `jacobi_matrix()` + quadrature 权重计算参考结果，证明公共 `integral()` 与统一契约一致。

### 验证命令

```bash
pytest -q tests/mesh/unit/test_entity_integral.py tests/mesh/unit/schema/test_quadrilateral_schema.py tests/mesh/unit/schema/test_tetrahedron_schema.py
```

完成反馈必须说明各类实体的 Jacobian 测度公式和测试结果；不能只报告“测试通过”。

---

## 六、明确不分配的事项

以下内容在本轮不安排开发任务，原因是尚未形成可执行决策、范围过大或仍处于讨论状态：

1. BDF、INP、VTK 存取调用方式兼容：`known_problems.md` 明确标记为“待讨论”。
2. NodeMesh 及其它旧网格类特殊算法迁移：只有问题描述，没有本轮决策和验收边界。
3. 除 `from_box` 外的各种 `from_<...>` 构造方法：没有形成具体 API 决策。
4. HalfEdgeMesh、DartMesh、UniformMesh 迁移：决策是不进入新网格体系，仅整理原文件；不作为本轮 Mesh 开发任务。
5. 高阶形状、Polygon/Polyhedron：虽有“进一步实现即可”的方向性决策，但没有一天内可执行的具体范围、接口和验收标准，本轮暂不拆分任务。
6. 棱柱/棱锥 quad face 的额外复核、非仿射六面体高阶矩积分、多后端扩展、退化面处理策略：报告中尚无明确决策，暂不安排实现任务；可在 A/C/D 完成后作为下一轮决策输入。
7. `barycenter()` 物理质心语义：报告明确不作为本轮 Mesh bug。

## 七、完成标准

本轮任务只有在以下条件同时满足时才算完成：

- 负责人提交了代码修改和对应测试；
- 负责范围内的定向 pytest 实际通过；
- M07-A、M07-C、M07-D 各自负责的模块单元测试通过，或明确记录与本轮无关的既有失败；
- 没有把未决策事项擅自转化为实现行为；
- 完成反馈注明实际修改文件、测试命令、通过/失败结果和遗留问题。
