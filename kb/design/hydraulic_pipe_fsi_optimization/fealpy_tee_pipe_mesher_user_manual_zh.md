# FEALPy 三通管 FSI Mesher 用户手册

## 1. 功能概述

`TeePipeMesher` 用于生成三通管流固耦合（FSI）三维共形网格，提供两类输出：

- 主输出：`init_mesh()` 返回 `TetrahedronMesh`
- 附加输出：`mesh_data()` 返回完整 `mesh dict`（含边界/界面信息）

当前版本聚焦等径三通：

- 仅支持 `D_main == D_branch`
- 非等径三通（主管大于/小于支管）会明确报“暂不支持”


## 2. 依赖

- `gmsh`（Python 包）
- `fealpy`

示例环境：

```bash
cd src/fealpy
pip install -e ".[dev]"
pip install gmsh
```


## 3. 参数接口（工程语义）

核心参数：

- `D_main`：主管内径（mm）
- `D_branch`：支管内径（mm）
- `intersect_angle`：主支管夹角（度）
- `R_fillet`：交汇过渡圆角半径（mm）
- `L_in_ratio`：主管入口直段长度比例（相对 `D_main`）
- `L_out_ratio`：两根支管出口直段长度比例（相对 `D_branch`）
- `wall_thickness`：壁厚（mm）

可选网格参数：

- `mesh_size_global`：全局背景网格尺寸
- `mesh_size_junction`：交汇区局部网格尺寸
- `mesh_size_interface`：流固界面附近网格尺寸

当网格参数省略时，内部默认采用：

- `mesh_size_global = 0.1875 * D_branch`
- `mesh_size_junction = 0.125 * D_branch`
- `mesh_size_interface = 0.09375 * D_branch`

对于 `D_branch = 32`，默认即：

- `mesh_size_global = 6`
- `mesh_size_junction = 4`
- `mesh_size_interface = 3`


## 4. 当前约束与建议

- 仅支持等径：`D_main == D_branch`
- 当前稳定夹角范围：`45 <= intersect_angle <= 100`
- 需满足 `0 < R_fillet < D_branch / 2`

如果出现 OCC 圆角失败（fillet failure）：

1. 先减小 `R_fillet`
2. 再将 `intersect_angle` 调回 90° 附近验证
3. 适当增大 `mesh_size_global` 进行几何与网格调试


## 5. 主接口：`init_mesh()`

```python
from fealpy.mesher import TeePipeMesher

params = {
    "D_main": 32.0,
    "D_branch": 32.0,
    "intersect_angle": 90.0,
    "R_fillet": 12.25,
    "L_in_ratio": 5.0,
    "L_out_ratio": 10.0,
    "wall_thickness": 5.0,
    "mesh_size_global": 6.0,
    "mesh_size_junction": 4.0,
    "mesh_size_interface": 3.0,
}

mesher = TeePipeMesher(params)
mesh = mesher.init_mesh()
```

返回类型：

- `fealpy.mesh.TetrahedronMesh`

区域标签：

- `mesh.celldata["region"]` 保存体单元物理区域编号（`fluid`/`solid`）


## 6. 附加接口：`mesh_data()`

```python
mesh_dict = mesher.mesh_data()
```

`mesh_dict` 主要字段：

- `node`：网格节点坐标
- `tetra`：四面体单元
- `tetra_region`：体单元区域标记
- `boundary_tri`：边界三角形
- `boundary_tri_marker`：边界三角形物理组标记
- `interface_tri`：FSI 界面三角形
- `interface_adjacent_tet`：每个界面三角形邻接的两侧四面体（流体在前，固体在后）
- `interface_adjacent_region`：对应区域标记对（默认 `[fluid_tag, solid_tag]`）
- `physical_name_to_dimtag`：物理组名称到 `(维度, tag)` 的映射
- `physical_dimtag_to_name`：`(维度, tag)` 到物理组名称映射

常用物理组名：

- `fluid`
- `solid`
- `main_inlet`
- `branch1_outlet`
- `branch2_outlet`
- `fsi_interface`
- `outer_wall`
- `solid_main_inlet_end`
- `solid_branch1_end`
- `solid_branch2_end`


## 7. 最小可运行示例

```python
from fealpy.mesher import TeePipeMesher

params = {
    "D_main": 1.0,
    "D_branch": 1.0,
    "intersect_angle": 60.0,
    "R_fillet": 0.1,
    "L_in_ratio": 3.0,
    "L_out_ratio": 4.0,
    "wall_thickness": 0.05,
    "mesh_size_global": 0.6,
    "mesh_size_junction": 0.45,
    "mesh_size_interface": 0.35,
}

mesher = TeePipeMesher(params)
mesh = mesher.init_mesh()
mesh_dict = mesher.mesh_data()

print(mesh.node.shape, mesh.cell.shape)
print(sorted(mesh_dict["physical_name_to_dimtag"].keys()))
```


## 8. 常见问题

1. 报错 `unsupported now: D_main > D_branch` 或 `<`
   - 当前版本仅支持等径三通，请先使用 `D_main == D_branch`。

2. 报错 `intersect_angle must be between 45 and 100 degrees`
   - 当前版本为了保证 OCC 圆角稳定，限制了角度范围。

3. 报错 `failed to fillet tee junction with gmsh OCC`
   - 优先减小 `R_fillet`，并检查是否接近几何上限 `D_branch/2`。

4. 想获取 FSI 界面三角形
   - 使用 `mesh_data()["interface_tri"]`，不要从 `init_mesh()` 主返回直接取。
