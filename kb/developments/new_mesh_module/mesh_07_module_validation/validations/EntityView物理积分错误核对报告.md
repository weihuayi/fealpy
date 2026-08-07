# `EntityView` 物理积分错误核对报告

## 1. 结论

在当前提交 `c208a03f6` 上，对
`fealpy.mesh.schema.registry.SCHEMA_REGISTRY` 注册的 8 种实体进行了逐项核对。
核对内容包括：

- 常数积分是否等于实体测度；
- 一阶坐标矩是否符合解析值；
- 参考顶点是否映射到对应物理顶点；
- 张量积积分点轴是否满足 `EntityView.integral()` 的数据契约；
- `index` 是否只积分被选中的实体。

结果如下。

| schema | 核对结果 | 已确认的问题 |
|---|---:|---|
| `point` | 失败 | `integral()` 调用未实现的 `jacobi_matrix()`，抛出 `NotImplementedError` |
| `segment` | 通过 | 一维线段及嵌入三维空间的线段均通过常数与一阶矩检查 |
| `tri` | 失败 | 二维和嵌入三维空间的三角形积分均为正确值的 2 倍 |
| `quad` | 失败 | 当前循环节点顺序与 `jacobi_matrix()` 使用的张量积顺序不一致；结果随积分阶数变化，且 `index` 未生效 |
| `tet` | 失败 | 积分为正确值的 6 倍 |
| `prism` | 失败 | 常数积分为正确值的 2 倍；此外参考顶点与物理顶点的对应关系错误 |
| `pyramid` | 失败 | 底面循环节点顺序与几何形函数使用的张量积顺序不一致 |
| `hex` | 失败 | `bc_to_point()` 与 `jacobi_matrix()` 使用不同的积分点轴形状，`integral()` 直接抛出 `ValueError` |

因此，问题不能再表述为“仅确认三角形错误，四面体待核对”。四面体错误已经
确认，而且当前物理积分接口还存在多种相互独立的问题。也不能用一个统一的常数
因子修复全部实体。

直接涉及的实现文件为：

- `fealpy/mesh/schema/classic/base.py`
- `fealpy/mesh/schema/classic/point.py`
- `fealpy/mesh/schema/classic/quadrilateral.py`
- `fealpy/mesh/schema/classic/prism.py`
- `fealpy/mesh/schema/classic/pyramid.py`
- `fealpy/mesh/schema/classic/hexahedron.py`

## 2. 核对环境与判据

- FEALPy 提交：`c208a03f6`
- Python：`/home/edwin/feal-venv-py312/bin/python`
- backend：NumPy
- 日期：2026-07-27
- 测试积分阶数：主要使用 `q=5`，并用多个 `q` 检查结果是否稳定

最基本的必要条件是

$$
\operatorname{integral}(1)=\operatorname{measure}.
$$

对于仿射实体，一阶矩还应满足解析几何结果。例如，顶点为
$(0,0)$、$(2,0)$、$(0,1)$ 的三角形面积为 1，并且

$$
\int_K x\,\mathrm{d}x=\frac{2}{3},
\qquad
\int_K y\,\mathrm{d}x=\frac{1}{3}.
$$

只有常数积分通过并不足以证明映射正确，因此棱柱还检查了参考顶点插值，四边形
和棱锥还检查了当前拓扑节点顺序下的一阶矩。

## 3. 实测结果

### 3.1 常数积分与一阶矩

下表均使用当前 schema 的节点顺序。四边形和棱锥底面使用循环顺序，而不是旧的
张量积顺序。

| 实体 | 解析测度 | `integral(1)`，`q=5` | 结果 |
|---|---:|---:|---:|
| 长度为 2 的线段 | 2 | 2 | 通过 |
| 面积为 1 的二维三角形 | 1 | 2 | 2 倍 |
| 面积为 1、嵌入三维的三角形 | 1 | 2 | 2 倍 |
| 面积为 1 的单位正方形 | 1 | 0.4724252182 | 错误 |
| 面积为 1.5 的梯形 | 1.5 | 0.8508600495 | 错误 |
| 单位参考四面体 | $1/6$ | 1 | 6 倍 |
| 直三棱柱 | $1/2$ | 1 | 2 倍 |
| 单位方锥 | $1/3$ | 0.1574750727 | 错误 |
| 单位六面体 | 1 | 抛出 `ValueError` | 不可用 |
| 点实体 | 1 | 抛出 `NotImplementedError` | 不可用 |

四边形和棱锥的错误值依赖积分阶数。例如，单位正方形的常数积分为：

| `q` | 1 | 2 | 3 | 4 | 5 | 7 | 9 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `integral(1)` | 0 | 0.577350 | 0.430331 | 0.521267 | 0.472425 | 0.485269 | 0.490845 |

这说明它们不是遗漏一个固定参考测度因子，而是几何映射已经发生折叠。

### 3.2 三角形和四面体

三角形积分规则的权重和为 1，而仿射映射的 Jacobian 行列式绝对值为两倍物理
面积，因此当前实现得到

$$
\sum_q w_q |\det J|=2|K|.
$$

实测三角形的一阶矩同样全部放大 2 倍：

$$
\left(\int_K x,\int_K y\right)_{\mathrm{actual}}
=
\left(\frac{4}{3},\frac{2}{3}\right),
$$

而解析值为 $(2/3,1/3)$。嵌入三维空间的三角形得到相同的 2 倍错误。

四面体积分规则的权重同样归一化为 1，而仿射映射满足

$$
|\det J|=6|K|.
$$

因此当前常数积分和一阶矩均放大 6 倍。对单位参考四面体，

$$
|K|=\frac16,\qquad
\int_K x\,\mathrm{d}x
=\int_K y\,\mathrm{d}x
=\int_K z\,\mathrm{d}x
=\frac1{24},
$$

实际结果分别为 $1$ 和 $1/4$。

该问题在 `TriangleQuadrature`、`TetrahedronQuadrature` 以及切换到
`StroudQuadrature` 后均可复现，并非某一个积分公式的数据错误。

### 3.3 四边形

当前四边形拓扑采用循环节点顺序：

```text
0 -> 1 -> 2 -> 3 -> 0
```

`bc_to_point()` 在计算前使用 `[0, 1, 3, 2]` 将节点转成张量积顺序，但
`jacobi_matrix()` 直接使用原始 `cell`：

```python
# bc_to_point()
points = ctx.block.positions[quad[:, [0, 1, 3, 2]]]

# jacobi_matrix()
cell = ctx.sector.indices
J = bm.einsum("cim,qin->cqmn", node[cell], gphi)
```

因此函数值和 Jacobian 实际来自两个不同的几何映射。单位正方形被
`jacobi_matrix()` 解释为折叠四边形，`abs(det(J))` 又掩盖了折叠方向，最终
得到随积分阶数摆动并逐渐接近错误极限的面积。

此外，`jacobi_matrix()` 没有使用传入的 `index`。在含两个四边形的网格上执行

```python
view.integral(one, q=3, index=np.array([0], dtype=np.int32))
```

仍返回两个积分值，返回形状为 `(2,)`，而不是预期的 `(1,)`。

这个错误同样出现在三维网格中由拓扑构造出的四边形面上。单位立方体、直棱柱和
方锥的四边形面均可复现。

### 3.4 棱柱

棱柱存在两个独立问题。

第一，`PrismSchema.measure()` 已显式使用系数 `0.5`：

```python
return 0.5 * bm.einsum("q,cq->c", ws, l)
```

通用 `integral()` 没有同样的参考三角形测度，因而直三棱柱的常数积分为体积的
2 倍。

第二，`shape_function()` 的基函数顺序与 `_tp_points()` 的点顺序不一致。
对底面节点

```text
(0,0,0), (1,0,0), (0,1,0)
```

及对应的顶面节点，参考顶点映射得到：

| 参考顶点 | 预期物理点 | 实际物理点 |
|---|---|---|
| 三角形顶点 0、线段顶端 | `(0,0,1)` | `(1,0,1)` |
| 三角形顶点 1、线段底端 | `(1,0,0)` | `(0,0,1)` |

因此即使补上 `0.5`，一般函数积分仍不正确。`q=5` 时，当前一阶矩为

```text
Ix = 0.3456455717
Iy = 0.3241873679
Iz = 0.4974387658
```

仅乘 `0.5` 后仍不等于解析值 $(1/6,1/6,1/4)$。

### 3.5 棱锥

`PyramidSchema.SFace["quad"]` 和四个三角形侧面体现的是循环底面顺序
`0-1-2-3`，但 `geometry_shape_function()` 将底面节点解释为

```text
0 = (u0,v0), 1 = (u1,v0), 2 = (u0,v1), 3 = (u1,v1)
```

即张量积顺序。拓扑和几何形函数的节点语义不一致，导致循环编号的正常方锥被
映射为折叠实体。单位方锥在 `q=5` 时的常数积分仅为
`0.1574750727`，解析体积为 `1/3`。

以前用张量积底面顺序构造的孤立方锥可能得到正确积分，但这种节点顺序与当前
`SFace` 给出的边和侧面不一致，不能作为当前 schema 正确性的证据。

### 3.6 六面体

六面体 `bc_to_point()` 返回

```text
(NC, nq_w, nq_v, nq_u, GD)
```

而 `jacobi_matrix()` 返回展平后的

```text
(NC, NQ, GD, 3),  NQ = nq_u * nq_v * nq_w
```

通用 `integral()` 约定函数值和 Jacobian 使用同一个展平的 `NQ` 轴，因此单位
六面体在 `q=5` 时直接报错：

```text
ValueError: Size of label 'q' for operand 2 (125)
does not match previous terms (5).
```

这是积分点轴数据契约错误，不是积分精度问题。

### 3.7 点实体

`PointSchema` 提供了权重为 1 的 `PointQuadrature`，并且 `measure()` 返回 1，
但没有实现 `jacobi_matrix()`。通用 `ShapedEntitySchema.integral()` 无条件调用
`jacobi_matrix()`，所以点实体积分抛出 `NotImplementedError`。

## 4. 代码变更来源

提交 `4be573a96`（`fix: align box mesh and quadrilateral normals`）将
`ShapedEntitySchema.integral()` 从

```python
measure * normalized_quadrature_weights
```

改为逐积分点的 Jacobian 加权。这一方向对于非仿射实体是必要的，但当前实现没有
同时闭合以下契约：

1. 各参考实体的积分权重是按参考测度归一化，还是积分为 1；
2. 拓扑节点顺序、形函数顺序和 Jacobian 节点顺序必须一致；
3. 张量积积分点必须采用统一的展平规则；
4. 所有几何函数必须一致地应用 `index`。

该提交可直接解释点、三角形、四面体和棱柱常数积分相对于上一实现的回退，也在
同一提交中改变了四边形节点顺序。但棱柱参考顶点映射、棱锥节点语义和六面体
积分点轴还包含各 schema 内部的独立问题，不应全部归因于一个参考测度系数。

## 5. 影响范围

错误位于公共的 `EntityView.integral()` 与 schema 几何映射层，因此影响所有
调用该接口的上层模块。`EntityView.error()` 内部也调用 `integral()`，其范数会
继承相同的倍增、折叠或异常。

FVM 侧只通过网格公开接口请求物理域积分；三角形单元上的常数积分倍增已经通过
独立最小复现确认，并能解释相应误差计算和源项积分异常。这里不再展开 FVM
算法细节。

## 6. 建议的网格模块验收条件

修复后至少应为每种注册 schema 固化以下测试：

1. `integral(1)` 与 `measure()` 在同一容差内相等；
2. 仿射实体的一阶坐标矩与解析值一致；
3. 非仿射四边形的一阶矩与逐积分点 Jacobian 结果一致；
4. 所有参考顶点满足 Kronecker 节点插值，即映射到对应物理顶点；
5. `bc_to_point()`、`jacobi_matrix()` 和积分值统一为展平的 `NQ` 轴；
6. `index=[i]` 只返回一个实体的积分；
7. 三维单元的三角形面和四边形面重复执行同样的常数积分检查；
8. `EntityView.error(1, 0)` 等于总测度相应次幂下的解析范数。

建议用注册表驱动参数化测试，保证 `point`、`segment`、`tri`、`quad`、`tet`、
`prism`、`pyramid` 和 `hex` 不会再次只修复单一实体而遗漏其余实体。

## 7. 本报告的边界

- 已对当前注册的 8 种经典 schema 全部执行最小实测，不再保留“四面体待核对项”。
- 结果使用 NumPy backend；已确认的问题来自参考测度、节点排列和张量轴契约，
  不依赖特定线性求解器。
- 本报告没有检查高阶 Lagrange 实体或多边形实体，因为它们不在当前
  `SCHEMA_REGISTRY` 中。
- 本报告只记录和定位问题，没有修改 `mesh` 或 `fvm` 的实现。

## 附录 A：关键测试代码

以下脚本只依赖 FEALPy 和 NumPy。它覆盖全部 8 种已注册 schema，并额外检查：

- 二维和嵌入三维空间的三角形；
- 四边形积分对 `q` 的依赖以及 `index` 选择；
- 棱柱参考顶点映射；
- 三维单元生成的三角形面和四边形面。

在 FEALPy 仓库根目录执行：

```bash
/home/edwin/feal-venv-py312/bin/python -B - <<'PY'
import numpy as np

from fealpy.backend import bm
from fealpy.mesh import (
    HexahedronMesh,
    IntervalMesh,
    PrismMesh,
    PyramidMesh,
    QuadrangleMesh,
    TetrahedronMesh,
    TriangleMesh,
)


bm.set_backend("numpy")


def one(points):
    return np.ones(points.shape[:-1], dtype=points.dtype)


def coordinate(component):
    return lambda points: points[..., component]


def values(tensor):
    return np.asarray(tensor, dtype=float).tolist()


def audit_constant(name, mesh, expected_measure):
    view = mesh.Entity("cell")
    measure = np.asarray(view.measure(), dtype=float)
    record = {
        "name": name,
        "schema": view.schema.name,
        "measure": measure.tolist(),
        "expected_measure": expected_measure,
    }
    try:
        integral = np.asarray(view.integral(one, q=5), dtype=float)
        record["integral_1"] = integral.tolist()
        record["integral_over_measure"] = (integral / measure).tolist()
    except Exception as error:
        record["integral_error"] = (
            f"{type(error).__name__}: {error}"
        )
    print(record)


segment = IntervalMesh(
    np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    ),
    np.array([[0, 1]], dtype=np.int32),
)

triangle = TriangleMesh(
    np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    ),
    np.array([[0, 1, 2]], dtype=np.int32),
)

triangle_surface = TriangleMesh(
    np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    ),
    np.array([[0, 1, 2]], dtype=np.int32),
)

# from_box() 给出当前四边形 schema 使用的循环节点顺序。
quadrilateral = QuadrangleMesh.from_box(nx=1, ny=1)

tetrahedron = TetrahedronMesh(
    np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array([[0, 1, 2, 3]], dtype=np.int32),
)

prism = PrismMesh(
    np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int32),
)

# 方锥底面按当前 SFace 定义使用循环顺序 0-1-2-3。
pyramid = PyramidMesh(
    np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=np.float64,
    ),
    np.array([[0, 1, 2, 3, 4]], dtype=np.int32),
)

hexahedron = HexahedronMesh.from_box(
    box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    nx=1,
    ny=1,
    nz=1,
)

cases = (
    ("segment_3d", segment, [2.0]),
    ("triangle_2d", triangle, [1.0]),
    ("triangle_3d", triangle_surface, [1.0]),
    ("quadrilateral", quadrilateral, [1.0]),
    ("tetrahedron", tetrahedron, [1.0 / 6.0]),
    ("prism", prism, [0.5]),
    ("pyramid", pyramid, [1.0 / 3.0]),
    ("hexahedron", hexahedron, [1.0]),
)

print("=== constant integral matrix ===")
for case in cases:
    audit_constant(*case)

point_view = triangle.Entity("point")
try:
    point_integral = values(point_view.integral(one, q=5))
    print(
        {
            "name": "point",
            "measure": values(point_view.measure()),
            "integral_1": point_integral,
        }
    )
except Exception as error:
    print(
        {
            "name": "point",
            "measure": values(point_view.measure()),
            "integral_error": f"{type(error).__name__}: {error}",
        }
    )

print("=== analytic first moments ===")
moment_cases = (
    (
        "triangle",
        triangle.Entity("cell"),
        [2.0 / 3.0, 1.0 / 3.0],
    ),
    (
        "tetrahedron",
        tetrahedron.Entity("cell"),
        [1.0 / 24.0] * 3,
    ),
    (
        "prism",
        prism.Entity("cell"),
        [1.0 / 6.0, 1.0 / 6.0, 0.25],
    ),
)
for name, view, expected in moment_cases:
    actual = [
        values(view.integral(coordinate(d), q=5))
        for d in range(len(expected))
    ]
    print({"name": name, "actual": actual, "expected": expected})

print("=== quadrilateral q dependence and index ===")
quad_view = quadrilateral.Entity("cell")
for q in (1, 2, 3, 4, 5, 7, 9):
    print({"q": q, "integral_1": values(quad_view.integral(one, q=q))})

two_quads = QuadrangleMesh.from_box(nx=2, ny=1).Entity("cell")
selected = two_quads.integral(
    one,
    q=3,
    index=np.array([0], dtype=np.int32),
)
print(
    {
        "selected_shape": tuple(selected.shape),
        "expected_shape": (1,),
        "selected_values": values(selected),
    }
)

print("=== prism reference-vertex mapping ===")
triangle_vertex_0 = np.array([[1.0, 0.0, 0.0]])
triangle_vertex_1 = np.array([[0.0, 1.0, 0.0]])
segment_bottom = np.array([[1.0, 0.0]])
segment_top = np.array([[0.0, 1.0]])
prism_view = prism.Entity("cell")
print(
    {
        "reference": "triangle vertex 0, segment top",
        "actual": values(
            prism_view.bc_to_point(
                (triangle_vertex_0, segment_top)
            )
        ),
        "expected": [[[0.0, 0.0, 1.0]]],
    }
)
print(
    {
        "reference": "triangle vertex 1, segment bottom",
        "actual": values(
            prism_view.bc_to_point(
                (triangle_vertex_1, segment_bottom)
            )
        ),
        "expected": [[[1.0, 0.0, 0.0]]],
    }
)

print("=== face-sector constant integrals ===")
face_meshes = (
    ("tetrahedron", tetrahedron),
    ("prism", prism),
    ("pyramid", pyramid),
    ("hexahedron", hexahedron),
)
for parent_name, mesh in face_meshes:
    for face_view in mesh.Entities("face"):
        measure = np.asarray(face_view.measure(), dtype=float)
        try:
            integral = np.asarray(
                face_view.integral(one, q=3),
                dtype=float,
            )
            result = {
                "parent": parent_name,
                "face_schema": face_view.schema.name,
                "count": int(measure.size),
                "ratio_min": float(np.min(integral / measure)),
                "ratio_max": float(np.max(integral / measure)),
            }
        except Exception as error:
            result = {
                "parent": parent_name,
                "face_schema": face_view.schema.name,
                "count": int(measure.size),
                "integral_error": f"{type(error).__name__}: {error}",
            }
        print(result)
PY
```

当前提交上的关键输出应包括：

```text
segment_3d: integral_over_measure = [1.0]
triangle_2d: integral_over_measure ≈ [2.0]
triangle_3d: integral_over_measure ≈ [2.0]
quadrilateral: integral_over_measure ≈ [0.4724252182]
tetrahedron: integral_over_measure ≈ [6.0]
prism: integral_over_measure ≈ [2.0]
pyramid: integral_over_measure ≈ [0.4724252182]
hexahedron: ValueError
point: NotImplementedError
quadrilateral selected_shape = (2,), expected_shape = (1,)
```

修复后的验收目标不是维持这些输出，而是使全部常数积分比值为 1、所有一阶矩和
参考顶点映射符合解析值，并使六面体与点实体不再抛出异常。
