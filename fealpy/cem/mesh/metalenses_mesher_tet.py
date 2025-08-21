# 整体网格生成
import gmsh
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh
import json


class MetalensesMesherTet:
    """
    create a tetrahedral mesh for metalenses.

    Parameters:
        params: dict
            A dictionary containing parameters for the metalenses, including:
            glass_size: float
                Size of the base square.
            glass_height: float
                Height of the glass layer(base square).
            air_layer_height: float
                Height of the air layer.
            bottom_pml_height: float
                Height of the bottom PML layer.
            top_pml_height: float
                Height of the top PML layer.
            antenna1_size: float
                Size of the first antenna.
            antenna1_height: float
                Height of the first antenna.
            antenna2_size: float
                Size of the second antenna.
            antenna2_height: float
                Height of the second antenna.
            antenna3_size: float
                Size of the third antenna.
            antenna3_height: float
                Height of the third antenna.
            antenna4_size: float
                Size of the fourth antenna.
            antenna4_height: float
                Height of the fourth antenna.
    """
    def __init__(self, params):
        self.glass_size = params["glass_size"]
        self.glass_height = params["glass_height"]
        self.air_layer_height = params["air_layer_height"]
        self.bottom_pml_height = params["bottom_pml_height"]
        self.top_pml_height = params["top_pml_height"]
        self.antenna1_size = params["antenna1_size"]
        self.antenna1_height = params["antenna1_height"]
        self.antenna2_size = params["antenna2_size"]
        self.antenna2_height = params["antenna2_height"]
        self.antenna3_size = params["antenna3_size"]
        self.antenna3_height = params["antenna3_height"]
        self.antenna4_size = params["antenna4_size"]
        self.antenna4_height = params["antenna4_height"]


    def generate_mesh(self, mesh_size=0.2):
        glass_size = self.glass_size
        glass_height = self.glass_height
        air_layer_height = self.air_layer_height
        bottom_pml_height = self.bottom_pml_height
        top_pml_height = self.top_pml_height
        antenna1_size = self.antenna1_size
        antenna1_height = self.antenna1_height
        antenna2_size = self.antenna2_size
        antenna2_height = self.antenna2_height
        antenna3_size = self.antenna3_size
        antenna3_height = self.antenna3_height
        antenna4_size = self.antenna4_size
        antenna4_height = self.antenna4_height

        # 初始化 GMSH
        gmsh.initialize()
        gmsh.model.add("metalenses")
        # 构造几何体
        base = gmsh.model.occ.addBox(-glass_size / 2, -glass_size / 2, -glass_height, glass_size, glass_size, glass_height)
        antenna1 = gmsh.model.occ.addBox(glass_size / 4 - antenna1_size / 2, glass_size / 4 - antenna1_size / 2, 0,
                                         antenna1_size, antenna1_size, antenna1_height)
        antenna2 = gmsh.model.occ.addBox(-glass_size / 4 - antenna2_size / 2, glass_size / 4 - antenna2_size / 2, 0,
                                         antenna2_size, antenna2_size, antenna2_height)
        antenna3 = gmsh.model.occ.addBox(-glass_size / 4 - antenna3_size / 2, -glass_size / 4 - antenna3_size / 2, 0,
                                         antenna3_size, antenna3_size, antenna3_height)
        antenna4 = gmsh.model.occ.addBox(glass_size / 4 - antenna4_size / 2, -glass_size / 4 - antenna4_size / 2, 0,
                                         antenna4_size, antenna4_size, antenna4_height)
        air_layer = gmsh.model.occ.addBox(-glass_size / 2, -glass_size / 2, 0,
                                          glass_size, glass_size, air_layer_height)
        bottom_plm_layer = gmsh.model.occ.addBox(-glass_size / 2, -glass_size / 2, -glass_height - bottom_pml_height,
                                                 glass_size, glass_size, bottom_pml_height)
        top_plm_layer = gmsh.model.occ.addBox(-glass_size / 2, -glass_size / 2, air_layer_height,
                                              glass_size, glass_size, top_pml_height)
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
        translation_x = [1, 0, 0, glass_size,
                         0, 1, 0, 0,
                         0, 0, 1, 0,
                         0, 0, 0, 1]
        translation_y = [1, 0, 0, 0,
                         0, 1, 0, glass_size,
                         0, 0, 1, 0,
                         0, 0, 0, 1]
        gmsh.model.mesh.setPeriodic(2, [65, 60, 58, 70], [64, 59, 54, 69], translation_x)
        gmsh.model.mesh.setPeriodic(2, [67, 62, 57, 72], [66, 61, 55, 71], translation_y)

        # 划分物理组
        fragmented_volumes = gmsh.model.occ.getEntities(dim=3)
        volume_to_physical = {}
        for i, (dim, tag) in enumerate(fragmented_volumes):
            gmsh.model.addPhysicalGroup(dim, [tag], tag)
            volume_to_physical[tag] = tag  # 使用 tag 本身作为标识符（你也可以用自定义值）
        gmsh.model.occ.synchronize()

        # 网格生成
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size * glass_size)
        gmsh.model.mesh.generate(3)
        # 获取周期边界的对应点
        origin_face_tag_x = [65, 60, 58, 70]
        periodic_node_pairs_x = {}
        for face in origin_face_tag_x:
            master_tag, origin_node_tag, master_node_tag, _ = gmsh.model.mesh.getPeriodicNodes(2, face)
            if origin_node_tag is not None:
                for i in range(len(origin_node_tag)):
                    periodic_node_pairs_x[int(origin_node_tag[i]-1)] = int(master_node_tag[i]-1)
        origin_face_tag_y = [67, 62, 57, 72]
        periodic_node_pairs_y = {}
        for face in origin_face_tag_y:
            master_tag, origin_node_tag, master_node_tag, _ = gmsh.model.mesh.getPeriodicNodes(2, face)
            if origin_node_tag is not None:
                for i in range(len(origin_node_tag)):
                    periodic_node_pairs_y[int(origin_node_tag[i]-1)] = int(master_node_tag[i]-1)
        # 获取所有节点坐标
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = bm.array(node_coords, dtype=bm.float64).reshape(-1, 3)
        # 获取所有三维单元（wedge）：
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
        cells = bm.array(elem_node_tags, dtype=bm.int64).reshape(-1, 4) - 1  # Gmsh 节点是从 1 开始编号
        # 获取所有元素和其物理组
        physical_groups = gmsh.model.getPhysicalGroups(3)
        cell_to_physical = {}
        for dim, phys_tag in physical_groups:
            etypes, etags, _ = gmsh.model.mesh.getElements(dim, phys_tag)
            for e in etags:
                for elem_id in e:
                    cell_to_physical[elem_id] = phys_tag
        # 创建 TetrahedronMesh 对象
        tet_mesh = TetrahedronMesh(points, cells)
        tet_mesh.celldata['domain'] = bm.array([cell_to_physical.get(i + 1, -1) for i in range(len(cells))],
                                               dtype=bm.int64)
        tet_mesh.meshdata['node_pairs'] = [periodic_node_pairs_x, periodic_node_pairs_y]
        # 结束GMSH
        gmsh.finalize()

        return tet_mesh


if __name__ == "__main__":
    # 参数读取
    with open("../data/parameters.json", "r") as f:
        params = json.load(f)
    # glass_size = params["glass_size"]
    # glass_height = params["glass_height"]
    # air_layer_height = params["air_layer_height"]
    # bottom_pml_height = params["bottom_pml_height"]
    # top_pml_height = params["top_pml_height"]
    # antenna1_size = params["antenna1_size"]
    # antenna1_height = params["antenna1_height"]
    # antenna2_size = params["antenna2_size"]
    # antenna2_height = params["antenna2_height"]
    # antenna3_size = params["antenna3_size"]
    # antenna3_height = params["antenna3_height"]
    # antenna4_size = params["antenna4_size"]
    # antenna4_height = params["antenna4_height"]

    # 控制参数
    mesh_size = 0.2

    mesher = MetalensesMesherTet(params)
    tet_mesh = mesher.generate_mesh(mesh_size)
    tet_mesh.to_vtk(fname="../data/metalenses_tet_mesh.vtu")




















