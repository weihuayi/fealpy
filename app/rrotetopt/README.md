# rrotetopt

`rrotetopt` 是一个基于**半径比（radius ratio）能量函数**的单纯形网格优化程序包，当前包含：

- 二维三角形网格优化
- 曲面三角形网格优化
- 三维四面体网格优化
- 二维/三维各向异性网格优化
- L-BFGS 与 NLCG 两类无约束优化算法
- 可选的投影型预条件子
- 局部拓扑翻转（主要用于三维四面体及部分各向异性算例）

程序整体上分为三层：

1. **网格/算例构造层**：负责生成测试网格与几何模型；
2. **Problem 建模层**：负责定义半径比质量函数、梯度、边界投影、切向约束、预条件子矩阵等；
3. **优化器层**：负责执行 L-BFGS / NLCG 迭代与线搜索。

---

## 1. 目录结构

```text
rrotetopt/
├── MeshModel.py
├── AniMeshModel.py
├── opt_example2d.py
├── opt_example3d.py
├── optani_example2d.py
├── optani_example3d.py
├── runner_example.py
├── meshopt/
│   ├── TriMeshProblem.py
│   ├── TriSurfMeshProblem.py
│   ├── TetMeshProblem.py
│   ├── TriAniMeshProblem.py
│   ├── TetAniMeshProblem.py
│   ├── rropreconditioner.py
│   └── meshopt_runner.py
└── opt/
    ├── optimizer_base.py
    ├── PLBFGSAlg.py
    ├── PNLCGAlg.py
    ├── line_search.py
    ├── preconditioner.py
    └── __init__.py
```

---

## 2. 依赖环境

建议 Python 版本：`Python 3.10+`

主要依赖：

- `fealpy`
- `gmsh`
- `numpy`
- `matplotlib`

如果需要导出 `vtu` 文件进行 ParaView 可视化，还需要确保 FEALPy 对应的网格输出环境可正常工作。

一个简单的安装示例：

```bash
pip install numpy matplotlib gmsh
```

`fealpy` 请根据你的本地环境单独安装。

---

## 3. 各文件作用说明

### 3.1 顶层文件

#### `MeshModel.py`
用于构造**各向同性**测试网格与几何模型。

目前包含的主要函数有：

- `to_TetrahedronMesh()`：将 gmsh 数据转为 `TetrahedronMesh`
- `to_TriangleMesh()`：将 gmsh 数据转为平面 `TriangleMesh`
- `to_TriangleMesh_Surface()`：将 gmsh 数据转为曲面三角网格
- `unit_sphere_surface(h=0.1)`：单位球面三角网格
- `unit_sphere(h=0.1)`：单位球体四面体网格
- `LShape(h=0.05)`：三维 L 形区域四面体网格
- `intersect_spheres(h=0.1)`：12 个相交球构成的几何区域
- `square_hole(h=0.05)`：带圆孔的二维方形区域
- `triangle_domain()`：二维三角形区域

该文件主要供 `opt_example2d.py`、`opt_example3d.py` 和 `runner_example.py` 调用。

#### `AniMeshModel.py`
用于构造**各向异性**算例所需的基础网格。

目前包含：

- `to_TetrahedronMesh()`
- `to_TriangleMesh()`
- `unit_circle(h=0.05)`：二维单位圆区域
- `unit_square(h=0.05)`：二维单位方形区域
- `unit_sphere(h=0.1)`：三维单位球区域
- `unit_square3d(h=0.1)`：三维单位立方体区域

该文件主要供 `optani_example2d.py` 和 `optani_example3d.py` 调用。

#### `opt_example2d.py`
二维各向同性优化示例脚本，展示如何直接调用：

- `TriMeshProblem`：平面三角形网格优化
- `TriSurfMeshProblem`：曲面三角形网格优化
- `PLBFGS` / `PNLCG`：优化算法

内置示例函数：

- `test_triangle_domain()`：二维三角形区域优化
- `test_square_hole()`：带圆孔方形区域优化
- `test_unit_sphere_surf()`：单位球面曲面网格优化

同时包含质量直方图绘制函数 `show_mesh_quality()`。

#### `opt_example3d.py`
三维各向同性优化示例脚本，展示如何直接调用：

- `TetMeshProblem`：四面体网格优化
- `PLBFGS` / `PNLCG`

内置示例函数：

- `test_unit_sphere()`：单位球体四面体网格优化
- `test_LShape()`：L 形区域四面体网格优化
- `test_intersectsphere()`：12 相交球几何的四面体网格优化

同时包含：

- `show_mesh_quality()`：半径比质量统计图
- `show_angle()`：二面角统计图

#### `optani_example2d.py`
二维各向异性优化示例脚本。

内置示例函数：

- `test_unit_circle()`：单位圆区域各向异性优化
- `test_unit_square()`：单位方形区域各向异性优化

脚本中定义了不同的度量张量 `metric`，通过 `mtype` 选择不同各向异性场。

#### `optani_example3d.py`
三维各向异性优化示例脚本。

内置示例函数：

- `test_unit_sphere()`：单位球各向异性四面体网格优化
- `test_square3d()`：单位立方体各向异性四面体网格优化

同样通过 `mtype` 选择不同度量张量。

#### `runner_example.py`
一个更统一的**高层调用示例**，基于 `meshopt/meshopt_runner.py` 中的统一接口运行优化流程。

相较于 `opt_example3d.py` 中“手动搭建 problem + optimizer”的写法，这个脚本把：

- 约束组织
- 参数配置
- 优化器选择
- 预条件子挂载
- 拓扑翻转外循环
- 结果回写与导出

都统一封装到 runner 中，更适合后续扩展和整理成正式程序接口。

当前内置：

- `run_unit_sphere()`
- `run_lshape()`
- `run_unit_12sphere()`

如果后续准备把程序整理成正式包接口，建议优先基于这一套 runner 框架继续扩展。

---

### 3.2 `meshopt/` 目录

#### `meshopt/TriMeshProblem.py`
二维**平面三角形网格**半径比优化问题。

主要功能：

- 定义二维三角形半径比质量函数
- 计算目标函数值与梯度
- 处理固定点、边界点、边界切向约束
- 构造预条件子所需矩阵
- 提供 `get_quality(mesh)` 用于质量统计

适用对象：二维平面三角形网格。

#### `meshopt/TriSurfMeshProblem.py`
二维**曲面三角形网格**优化问题。

主要功能：

- 面向嵌入三维空间中的曲面三角网格
- 支持边界投影 `Project`
- 支持边界曲线切向约束 `Tangent1d`
- 支持曲面法向约束 `Normal2d`
- 支持预条件子构造
- 提供 `get_quality(mesh)`

适用对象：球面、曲面离散网格等。

#### `meshopt/TetMeshProblem.py`
三维**四面体网格**半径比优化问题，是程序中的核心类之一。

主要功能：

- 定义四面体半径比目标函数与梯度
- 处理固定点、边界边点、边界面点的约束
- 构造预条件子矩阵
- 实现 `flipopt()` 局部拓扑翻转
- 提供 `get_quality(mesh)` 进行质量评估

适用对象：三维各向同性四面体网格优化。

#### `meshopt/TriAniMeshProblem.py`
二维**各向异性三角形网格**优化问题。

主要功能：

- 在给定度量张量场 `Metric(node)` 下计算各向异性质量函数
- 支持边界投影与边界切向约束
- 支持 `flipopt()` 拓扑优化
- 提供 `get_quality(mesh, metric)` 评估各向异性质量

#### `meshopt/TetAniMeshProblem.py`
三维**各向异性四面体网格**优化问题。

主要功能：

- 在给定三维度量张量场下构造各向异性目标函数
- 支持边界面法向约束、边界边切向约束和边界投影
- 提供 `flipopt()` 做拓扑翻转
- 提供 `get_quality(mesh, metric)` 进行各向异性质量评估

#### `meshopt/rropreconditioner.py`
半径比优化问题对应的**投影型预条件子**实现。

当前主要类：

- `ProjectedCGPreconditioner`

主要思路：

- 从 problem 中提取预条件矩阵 `P` 与投影矩阵 `Pi`
- 构造投影线性算子
- 通过 CG 近似求解预条件方程
- 给 L-BFGS / NLCG 提供预条件作用 `apply()`

#### `meshopt/meshopt_runner.py`
统一的高层优化入口。

核心对象：

- `ConstraintSpec`：约束说明
- `MeshOptConfig`：优化配置
- `MeshOptimizerRunner`：统一执行器
- `optimize_mesh(...)`：外部调用入口

这个模块的作用是把 Problem 与优化器的装配逻辑从 example 中剥离出来，便于程序组织与后续复用。

---

### 3.3 `opt/` 目录

#### `opt/optimizer_base.py`
优化器基类与问题基类。

主要定义：

- `Problem`：统一封装目标函数、初值、容差、步长等参数
- `Optimizer`：优化器基类，提供函数/梯度调用计数接口

#### `opt/PLBFGSAlg.py`
预条件 L-BFGS 优化算法实现。

主要功能：

- L-BFGS 两层递推
- 支持预条件子 `Preconditioner`
- 调用 Wolfe 线搜索

#### `opt/PNLCGAlg.py`
预条件非线性共轭梯度法（NLCG）实现。

主要功能：

- 共轭方向更新
- 支持预条件子
- 调用 Wolfe 线搜索

#### `opt/line_search.py`
线搜索模块。

目前实现：

- `zoom(...)`
- `wolfe_line_search(...)`

用于 L-BFGS / NLCG 的步长选择。

#### `opt/preconditioner.py`
预条件子抽象接口。

核心类：

- `BasePreconditioner`

定义了：

- `setup()`
- `update()`
- `apply()`
- `scale()`
- `reset()`

供具体预条件子类继承实现。

#### `opt/__init__.py`
对外导出基础接口：

- `BasePreconditioner`
- `Problem`

---

## 4. Example 脚本如何调用

下面给出当前几个示例脚本的命令行调用方式。

---

### 4.1 `opt_example2d.py`

用于二维各向同性优化与曲面三角网格优化。

#### 命令格式

```bash
python opt_example2d.py --exam <example_name> --optmethod <LBFGS|NLCG> --p <0|1>
```

#### 参数说明

- `--exam`：选择算例
  - `tri`：二维三角形区域
  - `sq`：带圆孔方形区域
  - `sps`：单位球面三角网格
- `--optmethod`：优化算法
  - `LBFGS`
  - `NLCG`
- `--p`：是否启用预条件子
  - `0`：不使用预条件子
  - `1`：使用预条件子

#### 示例

```bash
python opt_example2d.py --exam tri --optmethod LBFGS --p 0
python opt_example2d.py --exam sq --optmethod NLCG --p 1
python opt_example2d.py --exam sps --optmethod LBFGS --p 1
```

#### 输出内容

通常会输出：

- 初始/优化后网格质量统计图（PNG）
- 优化前后网格的 `vtu` 文件
- 终端中的优化耗时与部分优化信息

---

### 4.2 `opt_example3d.py`

用于三维各向同性四面体网格优化。

#### 命令格式

```bash
python opt_example3d.py --exam <example_name> --optmethod <LBFGS|NLCG> --p <0|1>
```

#### 参数说明

- `--exam`：选择算例
  - `sp`：单位球体
  - `ls`：L 形区域
  - `tsp`：12 相交球区域
- `--optmethod`：优化算法
  - `LBFGS`
  - `NLCG`
- `--p`：是否启用预条件子
  - `0`：不使用
  - `1`：使用

#### 示例

```bash
python opt_example3d.py --exam sp --optmethod LBFGS --p 0
python opt_example3d.py --exam ls --optmethod LBFGS --p 1
python opt_example3d.py --exam tsp --optmethod NLCG --p 1
```

#### 输出内容

一般包括：

- 质量分布图
- 二面角统计图（部分算例）
- 优化前后 `vtu` 文件
- 控制台优化信息

---

### 4.3 `optani_example2d.py`

用于二维各向异性网格优化。

#### 命令格式

```bash
python optani_example2d.py --exam <cir|squ> --optmethod <LBFGS|NLCG> --mtype <metric_type>
```

#### 参数说明

- `--exam`：算例类型
  - `cir`：单位圆
  - `squ`：单位方形
- `--optmethod`：优化算法
  - `LBFGS`
  - `NLCG`
- `--mtype`：度量张量类型
  - `1`：使用脚本中定义的第一类度量场
  - `2`：使用脚本中定义的第二类度量场

#### 示例

```bash
python optani_example2d.py --exam cir --optmethod LBFGS --mtype 1
python optani_example2d.py --exam squ --optmethod NLCG --mtype 2
```

---

### 4.4 `optani_example3d.py`

用于三维各向异性四面体网格优化。

#### 命令格式

```bash
python optani_example3d.py --exam <sp|cube> --optmethod <LBFGS|NLCG> --mtype <metric_type>
```

#### 参数说明

- `--exam`：算例类型
  - `sp`：单位球
  - `cube`：单位立方体
- `--optmethod`：优化算法
  - `LBFGS`
  - `NLCG`
- `--mtype`：度量张量类型
  - `1`：第一类度量场
  - `2`：第二类度量场（当前主要在球算例中定义）

#### 示例

```bash
python optani_example3d.py --exam sp --optmethod LBFGS --mtype 1
python optani_example3d.py --exam sp --optmethod LBFGS --mtype 2
python optani_example3d.py --exam cube --optmethod NLCG --mtype 1
```

---

### 4.5 `runner_example.py`

这是更推荐的统一调用方式，尤其适合后续整理成正式接口。

#### 命令格式

```bash
python runner_example.py --exam <sp|ls|tsp> --optmethod <LBFGS|NLCG> --p <True|False>
```

#### 参数说明

- `--exam`：算例类型
  - `sp`：单位球
  - `ls`：L 形区域
  - `tsp`：12 相交球区域
- `--optmethod`：优化算法
  - `LBFGS`
  - `NLCG`
- `--p`：是否启用预条件子
  - `False`：不使用
  - `True`：使用

> 注意：该脚本中 `--p` 当前被定义为 `type=bool`。在命令行中传参时，通常更稳妥的做法是直接修改脚本默认值，或在后续将其改成 `0/1` 或 `store_true/store_false` 风格。

#### 示例

```bash
python runner_example.py --exam sp --optmethod LBFGS --p False
python runner_example.py --exam ls --optmethod LBFGS --p True
python runner_example.py --exam tsp --optmethod NLCG --p True
```

---

## 5. 如果想在代码中直接调用

除了命令行方式，也可以在 Python 脚本中直接调用。例如，推荐使用 `runner_example.py` 对应的统一接口：

```python
from meshopt.meshopt_runner import ConstraintSpec, MeshOptConfig, optimize_mesh
import MeshModel

mesh = MeshModel.unit_sphere(h=0.1)

constraints = ConstraintSpec(
    Project=project_to_boundary,
    Normal2d=normal_vector,
    isFixNode=is_fix_node,
    isBdFaceNode=is_bd_face_node,
)

config = MeshOptConfig(
    problem_type='tet',
    optimizer='LBFGS',
    use_preconditioner=True,
    precond_update_interval=3,
    precond_rtol=1e-2,
    precond_maxiter=100,
    step_length=1.0,
    fun_val_diff=1e-6,
    step_length_tol=1e-6,
    norm_grad_tol=1e-6,
    max_iters=200,
    num_grad=10,
    print_info=True,
    flip=True,
    max_outer_iters=50,
    vtk_name='result.vtu',
)

result = optimize_mesh(mesh, constraints, config)
mesh = result['mesh']
```

其中：

### `ConstraintSpec` 常用参数

- `Project`：边界回投函数
- `Tangent1d`：边界曲线切向投影函数
- `Normal2d`：边界曲面法向函数
- `Metric`：各向异性度量张量函数
- `isFixNode`：固定点布尔数组
- `isBdEdgeNode`：边界边点布尔数组
- `isBdFaceNode`：边界面点布尔数组
- `FixAllBoundary`：是否固定全部边界点

### `MeshOptConfig` 常用参数

- `problem_type`
  - `tri`
  - `trisurf`
  - `triani`
  - `tet`
  - `tetani`
- `optimizer`
  - `LBFGS`
  - `NLCG`
- `use_preconditioner`：是否启用预条件子
- `precond_update_interval`：预条件子更新频率
- `precond_rtol`：CG 相对残差容忍度
- `precond_maxiter`：CG 最大迭代步数
- `step_length`：初始步长
- `fun_val_diff`：函数值收敛阈值
- `step_length_tol`：步长容差
- `norm_grad_tol`：梯度范数容差
- `max_iters`：内层优化最大迭代次数
- `num_grad`：与梯度相关的算法参数
- `print_info`：是否打印优化过程信息
- `flip`：是否启用拓扑翻转
- `max_outer_iters`：外层“优化 + 翻转”循环最大次数
- `vtk_name`：结果导出文件名

---

## 6. 输出结果说明

程序运行后通常会生成以下结果：

1. **VTK/VTU 文件**：用于 ParaView 可视化；
2. **质量统计图**：展示优化前后半径比质量分布；
3. **角度统计图**：部分三维算例会输出二面角分布；
4. **控制台信息**：包括优化耗时、目标函数值、外层翻转次数等。

---

## 7. 建议的使用方式

如果你只是想快速跑通算例，建议按下面顺序使用：

1. 先运行 `opt_example2d.py` / `opt_example3d.py`，熟悉每个算例；
2. 再运行 `runner_example.py`，熟悉统一配置方式；
3. 后续若要扩展自己的几何区域或边界约束，优先参考：
   - `ConstraintSpec`
   - `MeshOptConfig`
   - `TetMeshProblem` / `TriMeshProblem` 等 Problem 类

如果你后续准备把该程序包整理成论文配套代码或开源仓库，建议继续把 `runner_example.py + meshopt_runner.py` 这一套接口作为主入口保留下来，而把旧的 example 脚本作为演示文件。

---

## 8. 后续可进一步完善的内容

当前 README 已覆盖“文件作用 + example 调用方式 + 参数说明”的基本需求。后续还可以继续补充：

- 安装说明（`fealpy` 版本、gmsh 版本）
- 不同算例的结果图片
- 各质量指标的数学定义
- 预条件子的原理简介
- 常见报错与调试建议
- 与论文章节或公式的对应关系

---

## 9. 一个更简短的项目说明

本程序包实现了基于半径比能量函数的二维/三维单纯形网格优化算法，支持各向同性与各向异性情形，提供 L-BFGS 与 NLCG 两类优化器，并支持边界投影、切向约束、预条件子以及局部拓扑翻转，可用于三角形网格、曲面网格和四面体网格的质量提升与 sliver 单元消除。
