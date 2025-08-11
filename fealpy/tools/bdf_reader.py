from ..backend import backend_manager as bm

from pyNastran.bdf.bdf import BDF


def read_bdf_mesh(file_path):
    # 初始化 BDF 对象
    bdf = BDF()

    # 读取 .bdf 文件，使用 punch=True 处理仅包含 Bulk Data 的文件
    bdf.read_bdf(file_path, punch=True)

    # 提取节点
    nodes = []
    for node_id, node in bdf.nodes.items():
        # node.xyz 包含 [x, y, z] 坐标
        nodes.append([node.xyz[0], node.xyz[1], node.xyz[2]])

    # 转换为数组
    node = bm.array(nodes, dtype=float)  # 形状: (n_nodes, 4)

    # 提取三角形和四边形单元
    cell = []
    for elem_id, elem in bdf.elements.items():
        if elem.type == 'CTRIA3':
            # 三角形单元：node1, node2, node3
            cell.append([elem.nodes[0]-1, elem.nodes[1]-1, elem.nodes[2]-1])
        elif elem.type == 'CQUAD4':
            # 四边形单元：node1, node2, node3, node4
            cell.append([elem.nodes[0]-1, elem.nodes[1]-1, elem.nodes[2]-1, elem.nodes[3]-1])


    return node, cell