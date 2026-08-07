# FEALPy 弯管 FSI Mesher 用户手册

## 1. 功能概述

`ElbowPipeMesher` 用于生成弯管流固耦合（FSI）三维共形网格，提供两类输出：

- 主输出：`init_mesh()` 返回 `TetrahedronMesh`
- 附加输出：`mesh_data()` 返回完整 `mesh dict`（含边界/界面信息）

主输出仅包含体网格与区域标签，适合 FEALPy 常规求解流程；附加输出适合后续 FSI 接口处理。



## 2. 依赖

- `gmsh`（Python 包）
- `fealpy`

示例环境：

```bash
cd src/fealpy
pip install -e ".[dev]"
pip install gmsh
```



## 3. 参数接口

主参数（工程语义）：

- `D`：内径，单位与几何建模一致（例如 mm）。
- `bend_angle`：弯角（度），范围 `0 < bend_angle < 180`。
- `R_bend_inner`：内弯半径相对 `D` 的比例（无量纲）。实际内弯半径为 `R_bend_inner * D`。
- `L_in_ratio`：入口直管长度比例（相对 `D`）。实际入口长度为 `L_in_ratio * D`。
- `L_out_ratio`：出口直管长度比例（相对 `D`）。实际出口长度为 `L_out_ratio * D`。
- `wall_thickness`：壁厚，单位与 `D` 相同。

可选网格参数：

- `mesh_size_global`：全局背景网格尺寸。
- `mesh_size_bend`：弯头区域局部网格尺寸。
- `mesh_size_interface`：流固界面附近局部网格尺寸。

当可选网格参数省略时，内部自动采用：

- `mesh_size_global = 0.3 * D`
- `mesh_size_bend = 0.2 * D`
- `mesh_size_interface = 0.15 * D`



## 4. 主接口：`init_mesh()`

```python
from fealpy.mesher import ElbowPipeMesher

mesher = ElbowPipeMesher(
    {
        "D": 25.0,
        "bend_angle": 90.0,
        "R_bend_inner": 1.5,
        "L_in_ratio": 5.0,
        "L_out_ratio": 10.0,
        "wall_thickness": 5.0,
    }
)

mesh = mesher.init_mesh()
```

返回类型：

- `fealpy.mesh.TetrahedronMesh`

区域标签：

- `mesh.celldata["region"]` 保存体单元物理区域编号（fluid/solid 对应的 physical tag）



## 5. 附加接口：`mesh_data()`

```python
mesh_dict = mesher.mesh_data()
```

`mesh_dict` 主要字段：

- `node`：网格节点坐标；
- `tetra`：四面体网格单元；
- `tetra_region`：单元所属区域（1 为流体区域，2 为固体区域）；
- `boundary_tri`：边界三角形单元；
- `boundary_tri_marker`：边界三角形对应的物理组标记（与 `physical_name_to_dimtag` 对应）；
- `interface_tri`：FSI 界面的三角形单元；
- `interface_adjacent_tet`：每个界面三角形邻接的两个四面体单元索引（流体单元在前，固体单元在后）；
- `interface_adjacent_region`：与 `interface_adjacent_tet` 对应的区域标记对（默认 `[fluid_tag, solid_tag]`）；
- `physical_name_to_dimtag`：物理组名称到 `(维度, 物理组标签)` 的映射；
- `physical_dimtag_to_name`：`(维度, 物理组标签)` 到物理组名称的反向映射。

其中常用物理组名：

- `fluid`：流体体域；
- `solid`：固体体域；
- `inlet`：流体入口端面；
- `outlet`：流体出口端面；
- `fsi_interface`：流固耦合界面；
- `outer_wall`：固体外壁；
- `solid_inlet_end`：固体入口端环面；
- `solid_outlet_end`：固体出口端环面。



## 6. 最小可运行示例

```python
from fealpy.mesher import ElbowPipeMesher

params = {
    "D": 25.0,
    "bend_angle": 90.0,
    "R_bend_inner": 1.5,
    "L_in_ratio": 5.0,
    "L_out_ratio": 10.0,
    "wall_thickness": 5.0,
}

mesher = ElbowPipeMesher(params)
mesh = mesher.init_mesh()
mesh_dict = mesher.mesh_data()

print(mesh.node.shape, mesh.cell.shape)
print(mesh.celldata["region"].shape)
print(sorted(mesh_dict["physical_name_to_dimtag"].keys()))
```



## 7. 常见问题

1. `bend_angle` 报错  
检查是否满足 `0 < bend_angle < 180`。

2. 生成过慢  
优先增大 `mesh_size_global`、`mesh_size_bend`、`mesh_size_interface`。

3. 想获取 FSI 界面三角形  
使用 `mesh_data()["interface_tri"]`，不要从 `init_mesh()` 的主返回中直接取。

4. `R_bend_inner` 设置错误  
该参数是比例，不是绝对长度。例如 `R_bend_inner=1.5` 表示内弯半径为 `1.5 * D`。



## 8. 一键导出 VTU 示例脚本

脚本位置：

- `example/mesher/elbow_pipe_mesher_example.py`

运行示例：

```bash
python example/mesher/elbow_pipe_mesher_example.py \
  --D 25 --bend_angle 90 --R_bend_inner 1.5 \
  --L_in_ratio 5 --L_out_ratio 10 --wall_thickness 5 \
  --output_dir ./output_elbow
```

将生成：

- `elbow_pipe_tetra.vtu`
- `elbow_pipe_boundary_tri.vtu`
- `elbow_pipe_interface_tri.vtu`
