import numpy as np
from collections import defaultdict
from pyNastran.bdf.bdf import BDF


class DartMesh:
    def __init__(self, node, dart):
        self.node = node      # (NN, 3)
        self.dart = dart      # (ND, 7)

    @classmethod
    def from_surface_cells(cls, node: np.ndarray, cell: list):
        """
        node: (NN, 3)
        cell: list of (3,) or (4,) arrays
        """
        NC = len(cell)
        face_lens = [len(f) for f in cell]
        assert all(l in (3, 4) for l in face_lens), "Only triangles and quads supported."

        dart_list = []
        edge_dict = dict()  # edge -> edge_index
        edge_to_darts = defaultdict(list)

        edge_cnt = 0
        dart_idx = 0

        for c, face in enumerate(cell):
            nf = len(face)
            for i in range(nf):
                v0 = face[i]
                v1 = face[(i + 1) % nf]
                edge = tuple(sorted((v0, v1)))

                if edge not in edge_dict:
                    edge_dict[edge] = edge_cnt
                    edge_cnt += 1

                e = edge_dict[edge]

                dart = [-1] * 7
                dart[0] = v1        # vertex
                dart[1] = e         # edge id
                dart[2] = c         # face id
                dart[3] = c         # cell id
                dart_list.append(dart)
                edge_to_darts[edge].append(dart_idx)
                dart_idx += 1

        dart = np.array(dart_list, dtype=int)
        ND = dart.shape[0]

        # β1: next dart in face
        offset = 0
        for nf in face_lens:
            for i in range(nf):
                curr = offset + i
                next_ = offset + (i + 1) % nf
                dart[curr][4] = next_  # β1
            offset += nf

        # β2, β3: opposite darts on the same edge (one clockwise, one counterclockwise)
        for dart_indices in edge_to_darts.values():
            if len(dart_indices) == 2:
                d0, d1 = dart_indices
                dart[d0][5] = d1  # β2
                dart[d1][5] = d0
                dart[d0][6] = d1  # β3 = β2
                dart[d1][6] = d0
            elif len(dart_indices) == 1:
                d = dart_indices[0]
                dart[d][5] = -1  # boundary edge
                dart[d][6] = -1
            else:
                # warn: non-manifold edge
                for d in dart_indices:
                    dart[d][5] = -1
                    dart[d][6] = -1

        return cls(node, dart)

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

    # 转换为 NumPy 数组
    node = np.array(nodes, dtype=float)  # 形状: (n_nodes, 4)

    # 提取三角形和四边形单元
    cell = []
    for elem_id, elem in bdf.elements.items():
        if elem.type == 'CTRIA3':
            # 三角形单元：element_id, node1, node2, node3
            cell.append([elem.nodes[0], elem.nodes[1], elem.nodes[2]])
        elif elem.type == 'CQUAD4':
            # 四边形单元：element_id, node1, node2, node3, node4
            cell.append([elem.nodes[0], elem.nodes[1], elem.nodes[2], elem.nodes[3]])


    return node, cell


if __name__ == "__main__":
    # node = np.array([
    #     [0, 0, 0],  # v0
    #     [1, 0, 0],  # v1
    #     [1, 1, 0],  # v2
    #     [0, 1, 0],  # v3
    #     [2, 0, 0],  # v4
    # ], dtype=float)
    #
    # # 一个三角形 + 一个四边形
    # cell = [
    #     [0, 1, 3],  # triangle
    #     [1, 4, 2, 3],  # quad
    # ]
    #
    # dm = DartMesh.from_surface_cells(node, cell)
    #
    # print("Dart array (v, e, f, c, β1, β2, β3):")
    # print(dm.dart)
    file_path = './data/Sheet_Metal_20250717_Before_Opt_v2.bdf'
    node, cell = read_bdf_mesh(file_path)

    dm = DartMesh.from_surface_cells(node, cell)

    print("Dart array (v, e, f, c, β1, β2, β3):")
    print(dm.dart)
    print("Shape of nodes", dm.node.shape)
    print("Shape of darts:", dm.dart.shape)
