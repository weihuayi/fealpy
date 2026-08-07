# FEALPy | Execution Record | mesh_05b_ai_correction

- 执行时间：2026-06-09 14:49:31 +0800
- 执行人：AI Agent
- 本次范围：重试 `grad_lambda` 标准化实现、相关 schema 单测、旧类型依赖扫描与执行记录更新。

## 一、TDD / 调试过程摘要

### 1. RED 验证

新增测试文件：

- `tests/mesh/unit/schema/test_grad_lambda_standard.py`

最初 RED 命令：

```bash
uv run python -m pytest tests/mesh/unit/schema/test_grad_lambda_standard.py -q
```

失败符合预期，暴露的问题包括：

- `EntityView.grad_lambda()` 不接受 `ref`。
- `EdgeSchema.grad_lambda()` 等 schema 不接受 `ref` / `bcs`。
- `QuadrilateralSchema.grad_lambda()` 不接受 `bcs`。
- `HexahedronSchema` / `PrismSchema` / `PyramidSchema` 无统一 `grad_lambda` 实现或落到基类 `NotImplementedError`。

### 2. GREEN 实现

本次修改覆盖：

- `fealpy/mesh/schema/entity_schema.py`
  - 统一基类签名：`grad_lambda(ctx, index, bcs=None, *, ref=False)`。
  - 文档说明 `ref=False` 返回实际空间导数，`ref=True` 返回参考坐标导数。
- `fealpy/mesh/view/entity_view.py`
  - `EntityView.grad_lambda()` 转发 `index`、`bcs`、`ref`。
- `fealpy/mesh/schema/node.py`
  - 支持 `bcs` / `ref` 参数。
  - `quadrature_formula()` 补充 `qtype=None` 兼容参数。
- `fealpy/mesh/schema/edge.py`
  - `ref=True` 返回参考重心坐标单位阵。
  - `ref=False` 保持实际空间梯度。
- `fealpy/mesh/schema/triangle.py`
  - `ref=True` 返回参考重心坐标单位阵。
  - `ref=False` 保持 2D / 3D 实际梯度。
- `fealpy/mesh/schema/tetrahedron.py`
  - `ref=True` 返回参考重心坐标单位阵。
  - `ref=False` 保持 3D 实际梯度。
- `fealpy/mesh/schema/quadrilateral.py`
  - 支持 `bcs` 和 `ref`。
  - `bcs=None, ref=False` 保留既有中心点梯度语义与旧测试顺序。
  - `bcs!=None` 时返回带积分点维度的双线性节点形函数梯度。
- `fealpy/mesh/schema/hexahedron.py`
  - 增加双线性 / 三线性节点形函数梯度形式的 `grad_lambda()`。
- `fealpy/mesh/schema/prism.py`
  - 基于已有 `grad_shape_function()` / `first_fundamental_form()` 增加 `grad_lambda()`。
- `fealpy/mesh/schema/pyramid.py`
  - 基于已有 `geometry_grad_shape_function()` / `transform_grad()` 增加 `grad_lambda()`。
- `tests/mesh/unit/schema/test_hexahedron_schema.py`
  - 更新 handoff methods 断言，允许 `HexahedronSchema.grad_lambda`。

### 3. 中间失败与原因分析

第一次重试实现后运行：

```bash
uv run python -m pytest tests/mesh/unit/schema/test_grad_lambda_standard.py -q
```

结果：`1 failed, 4 passed`。

原因：`QuadrilateralSchema.grad_lambda()` 中 `bcs[0]` 有 2 个点而 `bcs[1]` 有 1 个点，构造参考导数时直接 `stack` 不同长度数组，触发：

```text
ValueError: all input arrays must have the same shape
```

修正：将 `v[:, 0]`、`v[:, 1]` broadcast 到 `u0.shape`。

随后单文件测试通过：

```text
5 passed in 0.52s
```

接着运行相关 schema 测试：

```bash
uv run python -m pytest tests/mesh/unit/schema -q
```

中间出现 4 个失败：

1. `HexahedronSchema` handoff method 白名单没有包含新增 `grad_lambda`。
   - 原因：测试白名单过窄，新增标准接口后应更新测试。
   - 修正：加入 `grad_lambda`。
2. `NodeSchema.quadrature_formula()` 不接受 `qtype`。
   - 原因：既有测试通过 view/schema 调用 `qtype=None`，签名不兼容。
   - 修正：补充 `qtype: str | None = None`。
3. `QuadrilateralSchema.grad_lambda(ctx, index)` 的默认返回节点顺序与既有测试冲突。
   - 原因：新实现默认中心点路径按内部 `[0,1,2,3]` 顺序返回，而既有测试要求沿用旧实现的 `[0,1,3,2]` 几何排序结果。
   - 修正：当 `bcs is None and ref is False` 时保留旧实现路径，避免破坏历史语义；带 `bcs` 的新路径保留标准化行为。
4. 新增测试中 `center_phys` 对 quad 的期望节点 2 / 3 顺序与既有旧语义不一致。
   - 原因：测试期望写错；旧语义及现有 schema 测试以 `[0,1,3,2]` 梯度顺序为准。
   - 修正：更新新增测试期望。

最终重新运行：

```bash
uv run python -m pytest tests/mesh/unit/schema -q
```

真实结果：

```text
........................................................................ [ 97%]
..                                                                       [100%]
74 passed in 0.60s
```

## 二、最终验证结果

已通过：

```bash
uv run python -m pytest tests/mesh/unit/schema/test_grad_lambda_standard.py -q
```

结果：

```text
5 passed in 0.52s
```

已通过：

```bash
uv run python -m pytest tests/mesh/unit/schema -q
```

结果：

```text
74 passed in 0.60s
```

## 三、旧类型依赖扫描

已生成：

- `kb/developments/new_mesh_module/mesh_05b_ai_correction/mesh_05b_ai_correction_old_type_dependency_report.md`

扫描结论：

- 命中文件总数：722。
- 对本轮 schema 基本算例不构成直接阻塞。
- 对后续“新网格最终替换旧网格”构成系统性迁移风险，需要单独后续任务处理。

## 四、剩余风险 / 待确认

1. `grad_lambda` 名称在 quad / hex / prism / pyramid 上仍有语义风险：本轮实现为低阶节点形函数参考/实际导数的过渡接口，不应误解为所有阶数形函数导数总接口。
2. `QuadrilateralSchema.grad_lambda(bcs=None, ref=False)` 为兼容既有测试保留旧中心点顺序；`bcs` 路径和 `ref=True` 路径表达参考坐标导数语义。后续若统一节点排序，应单独做破坏性迁移。
3. `normal()` / `tangent()` 返回形状统一尚未在本轮完全展开；本轮仅处理 `grad_lambda` 相关失败与执行记录。
4. 旧类型依赖数量很大，本轮只扫描记录，不迁移。
