from ..backend import backend_manager as bm

from pyNastran.bdf.bdf import BDF



mesh_type_map = {
    'CTRIA3': 'triangle',
    'CQUAD4': 'quadrangle',
    'CTETRA': 'tetrahedron',
    'CHEXA': 'hexahedron'
}


def read_bdf_mesh(file_path):
    # 初始化 BDF 对象
    bdf = BDF()

    # 读取 .bdf 文件，使用 punch=True 处理仅包含 Bulk Data 的文件
    try:
        bdf.read_bdf(file_path, punch=False)
    except Exception as e:
        try:
            # 如果读取失败，尝试不使用 punch
            bdf.read_bdf(file_path, punch=True)
        except Exception as e:
            raise RuntimeError(f"Failed to read BDF file: {file_path}, Error: {e}. "
                               f"try to delete the fist line with 'BEGIN BULK' "
                               f"and the last line with 'ENDDATA' in the file.")

    # 提取节点
    nodes = []
    nodes_id = []
    for node_id, node in bdf.nodes.items():
        nodes_id.append(node_id)
        # node.xyz 包含 [x, y, z] 坐标
        nodes.append([node.xyz[0], node.xyz[1], node.xyz[2]])

    # 转换为数组
    nodes = bm.array(nodes, dtype=float)  # 形状: (n_nodes, 4)

    # 提取三角形和四边形单元
    cells = {}
    cells_id = {}
    for elem_id, elem in bdf.elements.items():
        cells_id.setdefault(mesh_type_map[elem.type], []).append(elem_id)
        cells.setdefault(mesh_type_map[elem.type], []).append(elem)

    for k, v in cells.items():
        cells_id[k] = bm.array(cells_id[k], dtype=bm.int64)
        cells[k] = bm.array([node.node_ids for node in v], dtype=bm.int64)


    return nodes, nodes_id, cells, cells_id