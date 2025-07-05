# DRAFT
# DON'T USE THIS FILE
# NEED TO BE REWRITTEN
# GEOMETRIC PARAMETERS NEED TO BE UNIFIED
import gmsh
import meshio
import numpy as np
import sys
from fealpy.utils import timer


tmr = timer()
next(tmr)
gmsh.initialize()
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.Binary", 0)  # ASCII 格式
gmsh.clear()

# 获取角度
def rotate_angle(p):
    x = p[..., 0]
    y = p[..., 1]
    theta = np.pi / Lambda * (f - np.sqrt(x**2 + y**2 + f**2))
    return theta

# modify
name = "metalenses"
is_to_vtu = True  # 是否转换为 VTU 格式
layer_num = 2
length = 350
height = 20
# 参数定义
f = 60  # 设计焦距
Lambda = 0.98  # 入射光波长
# 几何参数
circle_radius = 20  # 需要生成立柱的圆的半径
# circle_radius = 5  # 需要生成立柱的圆的半径
periodic_unit_square_size = 0.4  # 周期单元正方形的边长
spacing = 0.4  # 相邻周期单元正方形的中心间距
box_length = 0.24  # 纳米柱的长度
box_width = 0.12  # 纳米柱的宽度
box_height = 0.6  # 纳米柱的高度
box_mesh_size = 0.5  # 纳米柱网格尺寸
base_size = 45  # 衬底正方形的长度
# base_size = 15  # 衬底正方形的长度
base_height = 0.1  # 衬底的高度
base_mesh_size = 1  # 衬底网格尺寸
# 网格参数
lc = 0.5  # 衬底处较大网格尺寸
box_lc = lc / 10  # 纳米柱处较小网格尺寸

gmsh.model.add(name)

print("开始创建几何模型...")
# 基底
gmsh.model.occ.addPoint(-base_size / 2, -base_size / 2, 0, base_mesh_size, 1)
gmsh.model.occ.addPoint(base_size / 2, -base_size / 2, 0, base_mesh_size, 2)
gmsh.model.occ.addPoint(base_size / 2, base_size / 2, 0, base_mesh_size, 3)
gmsh.model.occ.addPoint(-base_size / 2, base_size / 2, 0, base_mesh_size, 4)

gmsh.model.occ.addLine(1, 2, 1)
gmsh.model.occ.addLine(2, 3, 2)
gmsh.model.occ.addLine(3, 4, 3)
gmsh.model.occ.addLine(4, 1, 4)

gmsh.model.occ.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.occ.addPlaneSurface([1], 1)
gmsh.model.occ.synchronize()

# 添加纳米柱
def add_rectangle(x, y, z, dx, dy, lc, tag=None):
    p1 = gmsh.model.occ.addPoint(x - dx / 2, y - dy / 2, z, lc)
    p2 = gmsh.model.occ.addPoint(x + dx / 2, y - dy / 2, z, lc)
    p3 = gmsh.model.occ.addPoint(x + dx / 2, y + dy / 2, z, lc)
    p4 = gmsh.model.occ.addPoint(x - dx / 2, y + dy / 2, z, lc)

    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)

    c1 = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    if tag is None:
        tag = gmsh.model.occ.addPlaneSurface([c1])
        gmsh.model.occ.synchronize()
        return tag
    else:
        gmsh.model.occ.addPlaneSurface([c1], tag)
        gmsh.model.occ.synchronize()


# add_rectangle(0, 0, 0, box_length, box_width, box_mesh_size, 2)
# add_rectangle(1, 1, 0, box_length, box_width, box_mesh_size, 3)

# 生成周期单元的正方形中心坐标
square_centers = []
num_squares = int(circle_radius / spacing)
for i in range(-num_squares, num_squares + 1):
    for j in range(-num_squares, num_squares + 1):
        # 计算中心坐标
        center_x = i * spacing
        center_y = j * spacing
        # 检查周期单元的正方形是否完全在圆内（四个顶点都在给定圆内）
        if all(
                np.sqrt((center_x + dx) ** 2 + (center_y + dy) ** 2) <= circle_radius
                for dx in [-periodic_unit_square_size / 2, periodic_unit_square_size / 2]
                for dy in [-periodic_unit_square_size / 2, periodic_unit_square_size / 2]
        ):
            square_centers.append((center_x, center_y))

square_centers_array = np.array(square_centers)
angles = rotate_angle(square_centers_array)

# 添加立柱
boxes = []
for (cx, cy), angle in zip(square_centers_array, angles):
    # 添加 box，位于 (cx - box_length/2, cy - box_width/2, 0)
    x0 = cx - box_length / 2
    y0 = cy - box_width / 2
    z0 = 0
    # box = gmsh.model.occ.addBox(x0, y0, z0, box_length, box_width, box_height)
    box = add_rectangle(x0, y0, z0, box_length, box_width, box_mesh_size)
    # 以中心点旋转
    gmsh.model.occ.rotate([(2, box)], cx, cy, 0, 0, 0, 1, angle)
    boxes.append((2, box))
gmsh.model.occ.synchronize()
tmr.send("创建二维单独实体")
print(f"创建了 {len(boxes)} 个纳米柱")

# obj_out, map_out = gmsh.model.occ.fragment([(2, 1)], [(2, 2), (2, 3)])  # 将底板与纳米柱分割
# gmsh.model.occ.synchronize()

obj_out, map_out = gmsh.model.occ.fragment([(2, 1)], boxes)  # 将底板与纳米柱分割
gmsh.model.occ.synchronize()
tmr.send("分割合并二维实体")
print(f"分割合并完成")

# define slice (volume) geometry and mesh of slice dimension (depth)
dz = 1
gmsh.model.occ.extrude(
  obj_out[:-1], 0, 0, box_height, [0, layer_num], [0, 1], recombine=True)
gmsh.model.occ.extrude(
  obj_out, 0, 0, -base_height, [0, 1], [0, 1], recombine=True)
gmsh.model.occ.synchronize()
tmr.send("拉伸创建三维实体")
print("拉伸完成")

# mesh then save
gmsh.model.mesh.generate(3)
tmr.send("生成网格")
print("网格生成完成")

# ======================================================================
# 将 Gmsh 网格转换为 VTU 格式
# 查看当前网格中有哪些元素类型

# ======================================================================
if is_to_vtu:
    element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
    # 获取所有节点坐标
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    points = np.array(node_coords).reshape(-1, 3)
    # 获取所有三维单元（wedge）：
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
    wedge_cells = None
    for etype, etag_list, nodes_flat in zip(elem_types, elem_tags, elem_node_tags):
        et_name = gmsh.model.mesh.getElementProperties(etype)[0]
        if et_name == "Prism 6":
            wedge_cells = np.array(nodes_flat).reshape(-1, 6) - 1  # Gmsh 节点是从 1 开始编号
            break

    # 创建 meshio 网格并写入 .vtu
    if wedge_cells is not None:
        mesh = meshio.Mesh(
            points=points,
            cells=[("wedge", wedge_cells)]
        )
        meshio.write(f"../data/{name}_size{base_mesh_size}_lay{layer_num}_r{circle_radius}.vtu", mesh)
        print(f"✅ 成功写入 ../data/{name}_size{base_mesh_size}_lay{layer_num}_r{circle_radius}.vtu")
    else:
        print("❌ 没有找到三棱柱单元")
tmr.send("数据导出")
# gmsh.write(name + ".msh")

# view in GUI
# gmsh.fltk.run()
gmsh.finalize()
next(tmr)