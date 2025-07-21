import gmsh

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh, QuadrangleMesh


def translate_mesh(mesh,
                   node_pairs:list[dict],
                   translation_axes:int, translation_num:int):
    """
    Translate the mesh based on the periodic node pairs and translation axes.
    Maybe this method can be a static method for the Mesh class.

    Parameters:
        mesh: Mesh
            The original mesh.
        node_pairs: list[dict]
            A list of dictionary containing pairs of nodes to be translated,
            include the node pairs of axes needed to translated,
            and the node pairs of other axes.
        translation_axes: int
            The axes along which the translation will be applied.
            0 for x-axis, 1 for y-axis, 2 for z-axis.
        translation_num: int
            The number of translations to apply.
    """
    origin_node = mesh.node
    origin_cell = mesh.cell
    total_node = bm.copy(origin_node)
    total_cell = bm.astype(bm.copy(origin_cell), bm.int64)

    trans_node_pairs = bm.array(list(node_pairs[translation_axes].items()), dtype=bm.int64)
    origin_node_pairs = bm.copy(trans_node_pairs)
    del node_pairs[translation_axes]
    origin_other_node_pairs = [bm.array(list(p.items()), dtype=bm.int64) for p in node_pairs]

    domain = None
    if mesh.celldata['domain'] is not None:
        domain = mesh.celldata['domain']
        origin_domain = bm.copy(domain)

    num_unit_node = len(origin_node)
    num_interface_node = len(trans_node_pairs)
    translation_length = (origin_node[trans_node_pairs[0, 0], translation_axes]
                          - origin_node[trans_node_pairs[0, 1], translation_axes])
    is_interface_node = bm.zeros(num_unit_node, dtype=bm.bool)
    is_interface_node[trans_node_pairs[:, 1]] = True
    for i in range(translation_num):
        transform_node_map = bm.zeros(num_unit_node, dtype=bm.int64)
        transform_node_map[~is_interface_node] = bm.arange(
            num_interface_node+(i+1)*(num_unit_node-num_interface_node),
            num_interface_node+(i+2)*(num_unit_node-num_interface_node), dtype=bm.int64)
        transform_node_map[origin_node_pairs[:, 1]] = trans_node_pairs[:, 0]

        new_node = origin_node[~is_interface_node]
        new_node[:, translation_axes] += (i+1) * translation_length
        total_node = bm.concat([total_node, new_node], axis=0)

        new_cell = transform_node_map[origin_cell]
        total_cell = bm.concat([total_cell, new_cell], axis=0)

        # 更新平移方向周期点对
        new_node_pairs = bm.zeros_like(trans_node_pairs, dtype=bm.int64)
        new_node_pairs[:, 1] = trans_node_pairs[:, 0]
        new_node_pairs[:, 0] = transform_node_map[origin_node_pairs[:, 0]]
        trans_node_pairs = new_node_pairs

        # 更新其他方向的点对信息
        for i, node_pairs_dict in enumerate(node_pairs):
            new_node_pairs_i = bm.zeros_like(origin_other_node_pairs[i], dtype=bm.int64)
            new_node_pairs_i[:, 0] = transform_node_map[origin_other_node_pairs[i][:, 0]]
            new_node_pairs_i[:, 1] = transform_node_map[origin_other_node_pairs[i][:, 1]]
            for p in new_node_pairs_i:
                node_pairs_dict[p[0].item()] = p[1].item()
            node_pairs[i] = node_pairs_dict
        if domain is not None:
            domain = bm.concat([domain, origin_domain], axis=0)

    trans_node_pairs_dict = {}
    for i, p in enumerate(trans_node_pairs):
        trans_node_pairs_dict[p[0].item()] = origin_node_pairs[i, 1].item()
    node_pairs.insert(translation_axes, trans_node_pairs_dict)

    return total_node, total_cell, node_pairs, domain


if __name__ == '__main__':
    gmsh.initialize()
    gmsh.model.add("box_assemble")

    gmsh.model.occ.addBox(0, 0, 0, 3, 2, 1)
    gmsh.model.occ.synchronize()

    # 设置网格尺寸
    lc = 0.5
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
    gmsh.model.mesh.setSize([(0, 2)], lc / 10)
    # 设置周期边界
    affine_matrix_x = [1, 0, 0, 3,
                       0, 1, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1]
    affine_matrix_y = [1, 0, 0, 0,
                       0, 1, 0, 2,
                       0, 0, 1, 0,
                       0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(2, [2], [1], affine_matrix_x)
    gmsh.model.mesh.setPeriodic(2, [4], [3], affine_matrix_y)
    # 生成网格
    gmsh.model.mesh.generate(3)
    # 获取节点坐标与单元
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    node = bm.array(node_coords, dtype=bm.float64).reshape(-1, 3)
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(3)
    cell = bm.array(elem_node_tags, dtype=bm.int64).reshape(-1, 4) - 1
    tet_mesh = TetrahedronMesh(node, cell)
    # 获取周期边界信息
    origin_face_tag_x = [2]
    periodic_node_pairs_x = {}
    periodic_node_pairs_x_inv = {}
    for face in origin_face_tag_x:
        master_tag, origin_node_tag, master_node_tag, _ = gmsh.model.mesh.getPeriodicNodes(2, face)
        if origin_node_tag is not None:
            for i in range(len(origin_node_tag)):
                periodic_node_pairs_x[origin_node_tag[i] - 1] = master_node_tag[i] - 1
    origin_face_tag_y = [4]
    periodic_node_pairs_y = {}
    periodic_node_pairs_y_inv = {}
    for face in origin_face_tag_y:
        master_tag, origin_node_tag, master_node_tag, _ = gmsh.model.mesh.getPeriodicNodes(2, face)
        if origin_node_tag is not None:
            for i in range(len(origin_node_tag)):
                periodic_node_pairs_y[origin_node_tag[i] - 1] = master_node_tag[i] - 1
    node_pairs = [periodic_node_pairs_x, periodic_node_pairs_y]
    # 可视化
    # gmsh.fltk.run()
    # 结束
    gmsh.finalize()
    #
    # # 平移小单元，组装大网格
    # num_of_components = 3
    #
    # one_unit_node = tet_mesh.number_of_nodes()
    # num_of_interface_node = len(tet_mesh.meshdata["periodic_node_pairs"]["x_front"][0])
    # node_pairs_x = bm.array(list(tet_mesh.meshdata["periodic_node_pairs"]["x_front"][0].items()),
    #                         dtype=bm.int64)
    # node_pairs_y = bm.array(list(tet_mesh.meshdata["periodic_node_pairs"]["y_front"][0].items()),
    #                         dtype=bm.int64)
    # is_interface_node = bm.zeros(one_unit_node, dtype=bm.bool)
    # is_interface_node[node_pairs_x[:, 1]] = True
    # transform_node_map = bm.zeros(one_unit_node, dtype=bm.int64)
    # transform_node_map[~is_interface_node] = bm.arange(one_unit_node,
    #                                                    2*one_unit_node-num_of_interface_node,
    #                                                    dtype=bm.int64)
    # transform_node_map[node_pairs_x[:, 1]] = node_pairs_x[:, 0]
    # new_node = tet_mesh.node[~is_interface_node]
    # new_node[:, 0] += 3
    # new_node = bm.concat([tet_mesh.node, new_node], axis=0)
    # new_cell = transform_node_map[tet_mesh.cell]
    # new_cell = bm.concat([tet_mesh.cell, new_cell], axis=0)
    #
    # new_tet_mesh = TetrahedronMesh(new_node, new_cell)
    # new_node_pairs_x = bm.zeros_like(node_pairs_x, dtype=bm.int64)
    # new_node_pairs_x[:, 1] = node_pairs_x[:, 0]
    # new_node_pairs_x[:, 0] = transform_node_map[node_pairs_x[:, 0]]
    #
    # new_node_pairs_y = bm.zeros_like(node_pairs_y, dtype=bm.int64)
    # new_node_pairs_y[:, 0] = transform_node_map[node_pairs_y[:, 0]]
    # new_node_pairs_y[:, 1] = transform_node_map[node_pairs_y[:, 1]]
    #
    # periodic_node_pairs_x = tet_mesh.meshdata["periodic_node_pairs"]["x_front"][0]
    # periodic_node_pairs_y = tet_mesh.meshdata["periodic_node_pairs"]["y_front"][0]
    #
    # for i, p in enumerate(new_node_pairs_y):
    #     periodic_node_pairs_y[p[0]] = p[1]
    #
    # for p in periodic_node_pairs_y.items():
    #     print(p[0], "->", p[1])
    #     print(new_node[p[0]] - new_node[p[1]])
    new_node, new_cell, node_pairs, _ = translate_mesh(tet_mesh,
                                                    node_pairs,
                                                    translation_axes=0, translation_num=9)
    new_tet_mesh = TetrahedronMesh(new_node, new_cell)

    new_tet_mesh.to_vtk(fname='../data/assembled_mesh_x.vtu')
    new_node, new_cell, node_pairs, _ = translate_mesh(new_tet_mesh,
                                                    node_pairs,
                                                    translation_axes=1, translation_num=9)
    new_tet_mesh = TetrahedronMesh(new_node, new_cell)
    new_tet_mesh.to_vtk(fname='../data/assembled_mesh_y.vtu')

    # print(-1)

    # node = bm.array([[1, 1],
    #                  [2, 1],
    #                  [2, 2],
    #                  [0, 0],
    #                  [1, 0],
    #                  [2, 0],
    #                  [0, 1],
    #                  [1, 2],
    #                  [0, 2]], dtype=bm.float64)
    # cell = bm.array([[4, 5, 1, 0],
    #                  [0, 7, 8, 6],
    #                  [1, 2, 7, 0],
    #                  [4, 0, 6, 3]], dtype=bm.int64)
    # quad_mesh = QuadrangleMesh(node, cell)
    #
    # # node_pairs_x = bm.array([[5, 3],
    # #                          [1, 6],
    # #                          [2, 8]], dtype=bm.int64)
    # # node_pairs_y = bm.array([[8, 3],
    # #                          [7, 4],
    # #                          [2, 5]], dtype=bm.int64)
    # node_pairs_x = {5: 3,
    #                 1: 6,
    #                 2: 8}
    # node_pairs_y = {8: 3,
    #                 7: 4,
    #                 2: 5}
    # node_pairs = [node_pairs_x, node_pairs_y]
    #
    # new_node, new_cell, node_pairs = translate_mesh(quad_mesh,
    #                                                 node_pairs,
    #                                                 translation_axes=0, translation_num=9)
    # new_quad_mesh = QuadrangleMesh(new_node, new_cell)
    # new_node, new_cell, node_pairs = translate_mesh(new_quad_mesh,
    #                                                 node_pairs,
    #                                                 translation_axes=1, translation_num=9)
    # new_quad_mesh = QuadrangleMesh(new_node, new_cell)
    #
    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots()
    # new_quad_mesh.add_plot(ax)
    # new_quad_mesh.find_cell(ax, showindex=True)
    # new_quad_mesh.find_node(ax, showindex=True)
    # plt.show()



