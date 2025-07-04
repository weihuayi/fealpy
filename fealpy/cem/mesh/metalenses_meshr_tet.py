
# ====================================================================================
# 周期边界条件测试
# import gmsh
# from fealpy.backend import backend_manager as bm
# from fealpy.mesh import TetrahedronMesh
#
# # 初始化GMSH
# gmsh.initialize()
# gmsh.model.add("box")
#
# # 定义box的尺寸
# lx, ly, lz = 1.0, 1.0, 2.0
# is_to_vtu = True
# is_mesh = True
# show_inner = False
#
# # 创建box
# box1 = gmsh.model.occ.addBox(0, 0, 0, lx, ly, lz)
# box2 = gmsh.model.occ.addBox(0, 0, lz, lx, ly, lz/2)
# box = gmsh.model.occ.fragment([(3, box1)], [(3, box2)], removeObject=True)[0][0]
#
# # 同步几何
# gmsh.model.occ.synchronize()
#
# gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.5)
# gmsh.model.mesh.setSize([(0, 6)], 0.1)
#
# translation_x = [1, 0, 0, -lx,
#                         0, 1, 0, 0,
#                         0, 0, 1, 0,
#                         0, 0, 0, 1]
# translation_y = [1, 0, 0, 0,
#                         0, 1, 0, ly,
#                         0, 0, 1, 0,
#                         0, 0, 0, 1]
# translation_z = [1, 0, 0, 0,
#                         0, 1, 0, 0,
#                         0, 0, 1, lz,
#                         0, 0, 0, 1]
# gmsh.model.mesh.setPeriodic(2, [1, 7], [2, 8], translation_x)
# # gmsh.model.mesh.setPeriodic(2, [4], [3], translation_y)
#
# if is_mesh:
#     gmsh.model.mesh.generate(3)
#
# # 导出为VTK格式
# if is_to_vtu:
#     element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
#     # 获取所有节点坐标
#     node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
#     points = bm.array(node_coords).reshape(-1, 3)
#     # 获取所有三维单元（wedge）：
#     elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
#     cells = None
#     for etype, etag_list, nodes_flat in zip(elem_types, elem_tags, elem_node_tags):
#         et_name = gmsh.model.mesh.getElementProperties(etype)[0]
#         if et_name == "Tetrahedron 4":
#             cells = bm.array(nodes_flat).reshape(-1, 4) - 1  # Gmsh 节点是从 1 开始编号
#             break
#     if cells is not None:
#         # 创建 TetrahedronMesh 对象
#         tet_mesh = TetrahedronMesh(points, cells)
#         tet_mesh.to_vtk(fname="../data/box_mesh.vtu")
#         print("网格已保存为 VTU 格式: ../data/box_mesh.vtu")
#
#
#
#
# # 可视化
# if show_inner:
#     gmsh.fltk.run()
# # 结束GMSH
# gmsh.finalize()

# ============================================================================
# 区域标记测试
# import gmsh
# from fealpy.backend import backend_manager as bm
# from fealpy.mesh import TetrahedronMesh
#
#
# is_to_vtu = True  # 是否导出为VTU格式
# # 初始化 GMSH
# gmsh.initialize()
# gmsh.model.add("box_with_inner_box")
#
# # 设置全局容差
# gmsh.option.setNumber("Geometry.Tolerance", 1e-3)
#
# # 定义几何尺寸
# lx, ly, lz = 2.0, 2.0, 2.0  # 大 box 尺寸
# sx, sy, sz = 0.5, 0.5, 0.5  # 小 box 尺寸
#
# # 创建大 box（中心在 (0,0,0)）
# big_box = gmsh.model.occ.addBox(-lx/2, -ly/2, -lz/2, lx, ly, lz)
# print("Big box created with tag:", big_box)
#
# # 创建小 box（中心在 (0,0,0)）
# small_box1 = gmsh.model.occ.addBox(-sx, -sy, -sz, sx, sy, sz)
# print("Small box1 created with tag:", small_box1)
# small_box2 = gmsh.model.occ.addBox(-sx, -sy, sz, sx, sy, sz)
# print("Small box2 created with tag:", small_box2)
# small_box3 = gmsh.model.occ.addBox(-lx/2, -ly/2, -lz/2+lz, lx, ly, lz/2)
# print("Small box3 created with tag:", small_box3)
#
# # 执行 fragment 操作，确保界面一致
# obj_out, entity_map = gmsh.model.occ.fragment([(3, big_box)],
#                                               [(3, small_box1), (3, small_box2), (3, small_box3)],
#                                               removeObject=True)
# print("obj_out:", obj_out)
# print("entity_map:", entity_map)
#
# big_box = entity_map[0][0][1]  # 获取大 box 的新标签
#
# # 同步几何
# gmsh.model.occ.synchronize()
#
# # 调试：打印体视实体
# volumes = gmsh.model.getEntities(3)
# print("Volumes after fragment:", volumes)
#
#
# # 为 fragment 之后每个 volume 创建 Physical Group
# fragmented_volumes = gmsh.model.occ.getEntities(dim=3)
# volume_to_physical = {}
# for i, (dim, tag) in enumerate(fragmented_volumes):
#     gmsh.model.addPhysicalGroup(dim, [tag], tag)
#     volume_to_physical[tag] = tag  # 使用 tag 本身作为标识符（你也可以用自定义值）
#
# gmsh.model.occ.synchronize()
#
# # 网格生成
# gmsh.model.mesh.generate(3)
#
# # 获取所有元素和其物理组
# dim = 3
# element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim)
# physical_groups = gmsh.model.getPhysicalGroups(dim)
#
#
#
# # 输出单元所属 box 的标签
# # for elem_id, phys_tag in cell_to_physical.items():
# #     print(f"Element {elem_id} belongs to box (physical group) {phys_tag}")
# # gmsh.write("../data//box_with_inner.msh")
#
# if is_to_vtu:
#     element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
#     cell_to_physical = {}
#     # 获取所有节点坐标
#     node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
#     points = bm.array(node_coords).reshape(-1, 3)
#     # 获取所有三维单元（wedge）：
#     elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
#     cells = None
#     for etype, etag_list, nodes_flat in zip(elem_types, elem_tags, elem_node_tags):
#         et_name = gmsh.model.mesh.getElementProperties(etype)[0]
#         if et_name == "Tetrahedron 4":
#             cells = bm.array(nodes_flat).reshape(-1, 4) - 1  # Gmsh 节点是从 1 开始编号
#             # 建立单元到物理组的映射
#             for dim, phys_tag in physical_groups:
#                 entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
#                 for ent in entity_tags:
#                     etypes, etags, _ = gmsh.model.mesh.getElements(dim, ent)
#                     for e in etags:
#                         for elem_id in e:
#                             cell_to_physical[elem_id] = phys_tag
#             break
#     if cells is not None:
#         # 创建 TetrahedronMesh 对象
#         tet_mesh = TetrahedronMesh(points, cells)
#         tet_mesh.celldata['domain'] = bm.array([cell_to_physical.get(i+1, -1) for i in range(len(cells))], dtype=bm.int64)
#         tet_mesh.to_vtk(fname="../data/box_mesh.vtu")
#         print("网格已保存为 VTU 格式: ../data/box_mesh.vtu")
#
#
# # gmsh.fltk.run()
# # 结束GMSH
# gmsh.finalize()


# ========================================================================
# 整体网格生成
import gmsh
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh
import json


# 参数读取
with open("../data/parameters.json", "r") as f:
    params = json.load(f)
base_size = params["base_size"]
glass_height = params["glass_height"]
air_layer_height = params["air_layer_height"]
bottom_pml_height = params["bottom_pml_height"]
top_pml_height = params["top_pml_height"]
antenna1_size = params["antenna1_size"]
antenna1_height = params["antenna1_height"]
antenna2_size = params["antenna2_size"]
antenna2_height = params["antenna2_height"]
antenna3_size = params["antenna3_size"]
antenna3_height = params["antenna3_height"]
antenna4_size = params["antenna4_size"]
antenna4_height = params["antenna4_height"]

# 控制参数
is_mesh = True  # 是否生成网格
is_to_vtu = True  # 是否导出为VTU格式
is_show = False  # 是否显示GMSH GUI
mesh_size = 0.2  # 网格尺寸

# 初始化 GMSH
gmsh.initialize()
gmsh.model.add("metalenses")

# 构造几何体
base = gmsh.model.occ.addBox(-base_size/2, -base_size/2, -glass_height, base_size, base_size, glass_height)
antenna1 = gmsh.model.occ.addBox(base_size/4-antenna1_size/2, base_size/4-antenna1_size/2, 0,
                                 antenna1_size, antenna1_size, antenna1_height)
antenna2 = gmsh.model.occ.addBox(-base_size/4-antenna2_size/2, base_size/4-antenna2_size/2, 0,
                                 antenna2_size, antenna2_size, antenna2_height)
antenna3 = gmsh.model.occ.addBox(-base_size/4-antenna3_size/2, -base_size/4-antenna3_size/2, 0,
                                    antenna3_size, antenna3_size, antenna3_height)
antenna4 = gmsh.model.occ.addBox(base_size/4-antenna4_size/2, -base_size/4-antenna4_size/2, 0,
                                    antenna4_size, antenna4_size, antenna4_height)
air_layer = gmsh.model.occ.addBox(-base_size/2, -base_size/2, 0,
                                            base_size, base_size, air_layer_height)
bottom_plm_layer = gmsh.model.occ.addBox(-base_size/2, -base_size/2, -glass_height-bottom_pml_height,
                                            base_size, base_size, bottom_pml_height)
top_plm_layer = gmsh.model.occ.addBox(-base_size/2, -base_size/2, air_layer_height,
                                            base_size, base_size, top_pml_height)
gmsh.model.occ.synchronize()
# 几何体组合
antenna_with_air, _ = gmsh.model.occ.fragment([(3, air_layer)],
                                            [(3, antenna1), (3, antenna2), (3, antenna3), (3, antenna4)],
                                            removeObject=True)
air_with_base, _ = gmsh.model.occ.fragment(antenna_with_air,
                                            [(3, base)], removeObject=True)
total_shape, _ = gmsh.model.occ.fragment(air_with_base,
                                            [(3, bottom_plm_layer), (3, top_plm_layer)],
                                            removeObject=True)
gmsh.model.occ.synchronize()

# 设置周期边界条件
# 测试用，验证周期边界条件，实际可删除下面两行
# gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 2000)
# gmsh.model.mesh.setSize([(0, 77), (0, 65), (0, 73)], 200)
translation_x = [1, 0, 0, base_size,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1]
translation_y = [1, 0, 0, 0,
                        0, 1, 0, base_size,
                        0, 0, 1, 0,
                        0, 0, 0, 1]
gmsh.model.mesh.setPeriodic(2, [65, 60, 58, 70], [64, 59, 54, 69], translation_x)
gmsh.model.mesh.setPeriodic(2, [67, 62, 57, 72], [66, 61, 55, 71], translation_y)

# 划分物理组
#  创建 Physical Group
fragmented_volumes = gmsh.model.occ.getEntities(dim=3)
volume_to_physical = {}
for i, (dim, tag) in enumerate(fragmented_volumes):
    gmsh.model.addPhysicalGroup(dim, [tag], tag)
    volume_to_physical[tag] = tag  # 使用 tag 本身作为标识符（你也可以用自定义值）

gmsh.model.occ.synchronize()


if is_mesh:
    # 网格生成
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size*base_size)
    gmsh.model.mesh.generate(3)
    # 导出为VTU格式
    if is_to_vtu:
        # 获取所有节点坐标
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = bm.array(node_coords).reshape(-1, 3)
        # 获取所有三维单元（wedge）：
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
        cells = bm.array(elem_node_tags).reshape(-1, 4) - 1  # Gmsh 节点是从 1 开始编号
        # 获取所有元素和其物理组
        physical_groups = gmsh.model.getPhysicalGroups(3)
        cell_to_physical = {}
        for dim, phys_tag in physical_groups:
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
            for ent in entity_tags:
                etypes, etags, _ = gmsh.model.mesh.getElements(dim, ent)
                for e in etags:
                    for elem_id in e:
                        cell_to_physical[elem_id] = phys_tag
        # 创建 TetrahedronMesh 对象
        tet_mesh = TetrahedronMesh(points, cells)
        tet_mesh.celldata['domain'] = bm.array([cell_to_physical.get(i + 1, -1) for i in range(len(cells))],
                                               dtype=bm.int64)
        tet_mesh.to_vtk(fname="../data/metalenses_gmsh.vtu")
        print("网格已保存为 VTU 格式: ../data/metalenses_gmsh.vtu")


if is_show:
    gmsh.fltk.run()
# 结束GMSH
gmsh.finalize()




















