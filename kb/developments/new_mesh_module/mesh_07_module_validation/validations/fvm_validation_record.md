# FEALPy | mesh_07 | FVM 模块研究工作验证记录

- **模板版本**：v0.1
- **对应任务**：`mesh_07_module_validation`
- **记录标识**：`fvm_validation_record`
- **报告用途**：向新网格模块开发人员提交 FVM 迁移过程中观察到的可复现缺陷

> 本记录是 V&V 原始结果物，不是最终验收结论。所有判断仅覆盖本文件记录的 commit、环境、输入和用例。

## 一、成员与适用性

| 字段 | 填写内容 |
|---|---|
| 成员姓名 | FVM 模块开发成员（Edwin，AI 辅助复核） |
| member slug | `fvm` |
| 提交日期 | 2026-07-18 |
| 是否有基于 FEALPy 的现有研究工作 | 是 |
| 研究方向或项目简称 | 基于 FEALPy 的 cell-centred collocated 有限体积法，包含 Poisson、Stokes 和稳态不可压 Navier--Stokes 求解 |
| 记录状态 | `draft` |

### 不适用声明

不适用。本记录包含两个由 FVM 网格迁移工作触发的代表性验证用例。

## 二、验证基线与环境

| 字段 | 填写内容 |
|---|---|
| `develop` 完整 commit hash | `3413bffb9ac330fed0f5c9777382515eb3cfee0c` |
| 本地是否有附加修改 | 否；验证运行前受 Git 跟踪的工作树为空。验证后仅新增本 `_local` 报告，不修改 mesh/FVM 源码，不参与运行 |
| 操作系统 | Ubuntu 22.04.5 LTS，x86_64 |
| Python 版本 | Python 3.12.13，Clang 22.1.1 |
| FEALPy 安装或运行方式 | `/home/edwin/feal-venv-py312` 中以 editable 方式安装，源码位置 `/home/edwin/workplace/fealpy` |
| 后端与精度 | NumPy backend；输入和几何计算使用 `float64` |
| 关键依赖版本 | FEALPy 4.0.0、NumPy 2.5.1、SciPy 1.18.0 |
| 硬件信息（仅在相关时） | 与本次几何正确性验证无关 |

环境复现命令或依赖快照位置：

```bash
cd /home/edwin/workplace/fealpy
/home/edwin/feal-venv-py312/bin/python -B -m pip show fealpy
/home/edwin/feal-venv-py312/bin/python -B -c \
  "import sys, numpy, scipy; from fealpy.backend import bm; \
print(sys.version); print(numpy.__version__); print(scipy.__version__); print(bm.backend_name)"
```

## 三、代表性研究用例

### Case `fvm-01`：六面体网格四边形面的法向接口验证

| 字段 | 填写内容 |
|---|---|
| 用例目的 | 验证三维 cell-centred FVM 构造面面积向量所依赖的四边形面测度和法向是否可由新网格公开接口正确取得 |
| 选择理由 | 六面体控制体是三维 FVM 的基本网格类型；扩散、对流、散度和压力修正均直接依赖非零面法向 |
| 涉及 FEALPy 模块 | `mesh`、`fvm` |
| 涉及网格类型和关键操作 | `HexahedronMesh.from_box`；`Mesh.Entity("face")`；`measure()`；`normal()`；`face_normal()` |
| 输入来源 | 由官方 `HexahedronMesh.from_box` 生成一个单位立方体，不含外部数据 |
| 预期行为或结果 | 六个四边形面面积均为 1；每个面具有非零法向；`face_normal()` 返回 `(6, 3)` 且不抛出异常 |
| 预期依据 | 单位立方体解析几何；`EntityView.normal()` 和 `FEALPyMesh.face_normal()` 的公开接口说明 |
| 运行命令 | `python -B - <<'PY'`，完整自包含命令见下方；`python` 应指向开发人员待验证的 FEALPy 环境 |
| 运行状态 | `FAIL` |
| 开始与结束时间 | 2026-07-18；单次运行小于 2 s |

先激活开发人员待验证的 FEALPy Python 环境，再从 FEALPy 仓库根目录执行以下
自包含命令；无需预先创建临时文件：

```bash
python -B - <<'PY'
import numpy as np

from fealpy.backend import bm
from fealpy.mesh import HexahedronMesh

bm.set_backend("numpy")
mesh = HexahedronMesh.from_box(
    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], nx=1, ny=1, nz=1
)
face = mesh.Entity("face")

print("face indices:", np.asarray(face.indices).tolist())
print("face measure:", np.asarray(face.measure()).tolist())
print("EntityView.normal shape:", np.asarray(face.normal()).shape)
print("EntityView.normal:", np.asarray(face.normal()).tolist())

for name in ("face_normal", "face_unit_normal"):
    try:
        value = np.asarray(getattr(mesh, name)())
        print(name, value.shape, value.tolist())
    except Exception as exc:
        print(name, "ERROR", type(exc).__name__, str(exc))
PY
```

实际结果摘要：

```text
face measure: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
EntityView.normal shape: (6, 3)
EntityView.normal:
[[0.0, 0.0, 0.0], ..., [0.0, 0.0, 0.0]]
face_normal ERROR IndexError too many indices for array: array is 2-dimensional, but 3 were indexed
face_unit_normal ERROR IndexError too many indices for array: array is 2-dimensional, but 3 were indexed
```

预期与实际比较：

| 观察量 | 预期 | 实际 | 容差或判断规则 | 状态 |
|---|---|---|---|---|
| 面测度 | 六个面均为 1 | 六个面均为 1 | 绝对误差不大于 `1e-14` | match |
| `EntityView.normal()` | 每个面的法向范数严格大于 0 | 六个法向均为零 | 法向范数必须大于 0 | deviation |
| `face_normal()` | 返回 `(6, 3)` | `IndexError` | 不得抛出异常且返回形状符合文档 | deviation |
| `face_unit_normal()` | 返回六个有限单位向量 | `IndexError` | 不得抛出异常；范数应为 1 | deviation |

限制与未覆盖范围：

- 本用例只验证 NumPy backend 下的单位立方体，没有覆盖畸变、非平面四边形面或其它后端。
- 本用例没有运行 FVM 求解器；它验证的是求解器组装之前的必要几何量。
- 法向的全局正负朝向可由 FVM 根据 owner--neighbour 关系统一，本问题只要求返回几何上非零且形状契约一致的法向。

### Case `fvm-02`：非仿射四边形上的一阶矩积分验证

| 字段 | 填写内容 |
|---|---|
| 用例目的 | 验证新网格 `EntityView.integral()` 能否在畸变四边形上正确计算 cell average、源项和物理质心所需的一阶矩 |
| 选择理由 | FVM 的源项离散、制造解误差和控制体物理质心都依赖物理域积分；一般四边形不保证是仿射平行四边形 |
| 涉及 FEALPy 模块 | `mesh`、`fvm` |
| 涉及网格类型和关键操作 | `QuadrangleMesh(node, cell)`；`measure()`；`integral()`；`quadrature_formula()`；`jacobi_matrix()` |
| 输入来源 | 程序直接生成一个顶点为 `(0,0)`、`(2,0)`、`(1,1)`、`(0,1)` 的凸梯形；cell 使用新 schema 的张量积节点顺序 `[0,1,2,3]` |
| 预期行为或结果 | 面积为 `3/2`，`integral(x)=7/6`，`integral(y)=2/3` |
| 预期依据 | 多边形面积与一阶矩解析公式；同一 EntityView 的 Jacobian 逐积分点加权结果 |
| 运行命令 | `python -B - <<'PY'`，完整自包含命令见下方；`python` 应指向开发人员待验证的 FEALPy 环境 |
| 运行状态 | `FAIL` |
| 开始与结束时间 | 2026-07-18；单次运行小于 2 s |

先激活开发人员待验证的 FEALPy Python 环境，再从 FEALPy 仓库根目录执行以下
自包含命令；无需预先创建临时文件：

```bash
python -B - <<'PY'
import numpy as np

from fealpy.backend import bm
from fealpy.mesh import QuadrangleMesh

bm.set_backend("numpy")
node = np.array(
    [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    dtype=np.float64,
)
cell = np.array([[0, 1, 2, 3]], dtype=np.int32)
view = QuadrangleMesh(node, cell).Entity("cell")

qf = view.quadrature_formula(5)
bcs, ws = qf.get_quadrature_points_and_weights()
points = np.asarray(view.bc_to_point(bcs))
jacobian = np.asarray(view.jacobi_matrix(bcs))
det_jacobian = np.abs(np.linalg.det(jacobian))

public_x = np.asarray(view.integral(lambda p: p[..., 0], q=5))
public_y = np.asarray(view.integral(lambda p: p[..., 1], q=5))
jacobian_x = np.einsum("q,cq,cq->c", np.asarray(ws), det_jacobian, points[..., 0])
jacobian_y = np.einsum("q,cq,cq->c", np.asarray(ws), det_jacobian, points[..., 1])

print("measure:", np.asarray(view.measure()).tolist())
print("public integral x/y:", public_x.tolist(), public_y.tolist())
print("Jacobian integral x/y:", jacobian_x.tolist(), jacobian_y.tolist())
PY
```

实际结果摘要：

```text
measure: [1.5]
public integral x/y: [1.1249999999999998] [0.7499999999999998]
Jacobian integral x/y: [1.1666666666666665] [0.6666666666666666]
```

预期与实际比较：

| 观察量 | 预期 | 实际 | 容差或判断规则 | 状态 |
|---|---|---|---|---|
| `measure()` | `1.5` | `1.5` | 绝对误差不大于 `1e-12` | match |
| `integral(x)` | `7/6 = 1.1666666666666667` | `1.125` | 绝对误差不大于 `1e-12` | deviation |
| `integral(y)` | `2/3 = 0.6666666666666666` | `0.75` | 绝对误差不大于 `1e-12` | deviation |
| Jacobian 加权 `x` 一阶矩 | `7/6` | `1.1666666666666665` | 绝对误差不大于 `1e-12` | match |
| Jacobian 加权 `y` 一阶矩 | `2/3` | `0.6666666666666666` | 绝对误差不大于 `1e-12` | match |

限制与未覆盖范围：

- 本用例确认二维非仿射四边形上的问题，没有直接验证六面体、曲面四边形或其它后端。
- `barycenter()` 当前返回顶点平均值；由于其是否承诺物理面积质心仍存在语义解释空间，本记录不将该现象单独报告为 module bug。
- 本用例只验证积分数学正确性，没有评估性能。

## 四、发现的问题

| 临时报告编号 | Case | 问题摘要 | 类别候选 | 严重度候选 | 状态 | 支撑材料 |
|---|---|---|---|---|---|---|
| `R-fvm-01` | `fvm-01` | 官方六面体网格的四边形面法向为零，兼容层 `face_normal()`/`face_unit_normal()` 发生维数索引异常 | `module bug` | `high` | `reported` | 本记录 Case `fvm-01` 及问题 `R-fvm-01` |
| `R-fvm-02` | `fvm-02` | `EntityView.integral()` 在非仿射四边形上未按积分点 Jacobian 加权，静默产生错误一阶矩 | `module bug` | `high` | `reported` | 本记录 Case `fvm-02` 及问题 `R-fvm-02` |

### 问题 `R-fvm-01`

- **首次失败命令**：本记录 Case `fvm-01` 中的自包含 `python -B - <<'PY'` 命令。
- **预期行为**：官方工厂生成的非退化六面体，其六个四边形面应具有非零法向；兼容层接口应按文档返回 `(NF, GD)`。
- **实际行为**：面测度正确，但 `EntityView.normal()` 返回零；兼容层两个法向接口均抛出 `IndexError`。
- **错误类型或异常**：错误几何结果与接口返回形状不一致。
- **稳定复现情况**：2/2 次独立运行稳定复现。
- **最小复现状态**：available；代码完整包含于 Case `fvm-01`。
- **影响范围初判**：使用四边形作为 codimension-one 实体的三维网格，至少包括六面体；棱柱和棱锥的四边形面需由维护者进一步复核。依赖面法向的 FVM/FEM 算子均可能受影响。
- **临时规避方式**：上层可从 cell 局部面节点或 Jacobian 自行计算法向，但会重复 EntitySchema 几何逻辑，并需要另行处理全局面方向，不建议作为长期兼容方案。
- **敏感信息处理**：none。

关键 traceback 或差异：

```text
EntityView.normal(): shape=(6, 3), all entries are zero

File "fealpy/mesh/view/fealpy_api.py", line 649, in face_normal
    return block.normal(index=index)[:, 0, :]
IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed
```

最小复现步骤或复现阻塞说明：

1. 使用 `HexahedronMesh.from_box(..., nx=1, ny=1, nz=1)` 生成官方单位立方体网格。
2. 调用 `mesh.Entity("face").measure()` 和 `normal()`，观察面积正确但法向为零。
3. 调用 `mesh.face_normal()`，稳定得到二维返回值被三维索引的 `IndexError`。

根因候选，供网格模块开发人员复核：

- `TopologyBuilder` 从 root cell 的 `OFace` 构造四边形 sector，并保留代表面的原始环状节点顺序；
- `QuadrilateralSchema.normal()` 将 sector 中的四个节点再次按张量积节点顺序重排，可能把已有环状顺序变为交叉顺序，使两部分叉积抵消；
- `QuadrilateralSchema.normal()` 当前返回 `(NF, GD)`，但 `FEALPyMesh.face_normal()` 明确按 `(NF, 1, GD)` 使用。

### 问题 `R-fvm-02`

- **首次失败命令**：本记录 Case `fvm-02` 中的自包含 `python -B - <<'PY'` 命令。
- **预期行为**：物理域数值积分应在各积分点使用参考到物理映射的 Jacobian 测度因子，并正确积分一次多项式。
- **实际行为**：`measure()` 正确，但 `integral(x)` 和 `integral(y)` 均偏离解析一阶矩；使用同一 EntityView 的 `jacobi_matrix()` 逐点加权后恢复正确结果。
- **错误类型或异常**：无异常抛出，静默数值偏差。
- **稳定复现情况**：2/2 次独立运行稳定复现。
- **最小复现状态**：available；代码完整包含于 Case `fvm-02`。
- **影响范围初判**：非仿射张量积实体上的源项、误差、cell average 和由一阶矩构造的物理质心；六面体是否存在同类问题需由维护者复核。
- **临时规避方式**：调用者可以组合 `quadrature_formula()`、`jacobi_matrix()` 和 `bc_to_point()` 自行积分；该方式重复 mesh 的通用积分职责，不建议各上层模块分别实现。
- **敏感信息处理**：none。

关键 traceback 或差异：

```text
observable       expected               EntityView.integral()
integral(x)      1.1666666666666667     1.1249999999999998
integral(y)      0.6666666666666666     0.7499999999999998
```

最小复现步骤或复现阻塞说明：

1. 使用 `QuadrangleMesh(node, cell)` 构造 Case `fvm-02` 给出的凸梯形。
2. 调用 `EntityView.integral()` 计算坐标函数 `x`、`y` 的物理域积分。
3. 与解析多边形一阶矩及逐积分点 Jacobian 加权结果比较，稳定观察到偏差。

根因候选，供网格模块开发人员复核：

- `ShapedEntitySchema.integral()` 当前使用 `measure * reference_weight * value`；
- 该表达只在 Jacobian 测度因子为常量时成立；一般双线性四边形映射的 Jacobian 随参考坐标变化；
- 建议同时复核二维/三维、体单元/曲面单元的 Jacobian 测度因子定义，而不是只对本梯形增加特例。

## 五、成员级结论

| 项目 | 填写内容 |
|---|---|
| 已执行用例数 | 2 |
| PASS / FAIL / BLOCKED / INCONCLUSIVE | `0 / 2 / 0 / 0` |
| 报告问题数 | 2 |
| 当前研究用途判断 | `partially supported` |
| 判断适用边界 | `develop@3413bffb`；NumPy/float64；官方单位六面体和单个畸变四边形；只覆盖公开几何接口，不覆盖完整求解器 |
| 是否需要立即升级 | yes；建议在六面体 FVM 和畸变张量积网格的正式数值验证前完成独立复现和责任确认 |

结论说明：

本记录不支持“新 mesh 整体不兼容 FVM”的结论。拓扑关系和固定形状实体的公开视图足以让 FVM 开始适配；三角形、规则二维四边形和四面体路径不因本记录而被判定失败。但两个用例分别暴露了四边形面法向契约和非仿射实体积分的可复现问题。上层虽可利用节点、Jacobian 和关系接口临时重算，但这会把通用几何算法复制到 FVM。按照新网格模块的 EntitySchema 职责边界，建议由 mesh 模块复核和修复，并增加覆盖相应数学不变量的回归验证。

## 六、复核记录

| 字段 | 填写内容 |
|---|---|
| 成员自查 | FVM 模块开发成员（Codex 辅助），2026-07-18；两个最小用例均复跑，报告数值与当前源码一致 |
| 完整性检查 | Codex，2026-07-18；模板必填段落、commit、环境、命令、预期、实际、限制和临时报告编号均已填写 |
| 问题独立复现 | 待新网格模块开发人员执行；因此问题状态保持 `reported`，未填写 `reproduced` 或 `confirmed` |
| 复核备注 | 本记录没有修改 mesh/FVM 源码，没有将 FVM 旧接口调用或 `barycenter()` 语义争议登记为 mesh module bug |

## 附录：版本演进记录

- **v0.1**：
  - 变更时间：2026-07-18
  - 变更摘要：按 `mesh_07_module_validation_member_record_template.md` 首次记录 FVM 迁移发现的两个新网格几何问题。
