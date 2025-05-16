from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, TetrahedronMesh, IntervalMesh

from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

bm.set_backend('pytorch')


def get_bd_mesh(volume_mesh: TetrahedronMesh)->TriangleMesh:
    """从体网格获取边界网格"""
    bd_face_idx = volume_mesh.boundary_face_index()
    node = volume_mesh.node
    face = volume_mesh.face
    bd_node_idx = volume_mesh.boundary_node_index()
    bd_node = node[bd_node_idx]
    idx_map = bm.zeros(node.shape[0], dtype=bd_node_idx.dtype)
    idx_map[bd_node_idx] = bm.arange(bd_node_idx.shape[0], dtype=bd_node_idx.dtype)
    bd_cell = idx_map[face[bd_face_idx]]

    return TriangleMesh(bd_node, bd_cell)

def edge_sort(oriented_edge):
    """
    对边节点进行排序，使得相同的边在一起
    Parameters
    ----------
    oriented_edge : array
        边的节点索引，形状为 (NE, 2)，输入边的朝向一致
    Returns
    -------
    按照边连续的点索引
    """
    node_map = defaultdict(list)
    for idx, e in enumerate(oriented_edge):
        node_map[e[0].item()].append([idx, 0])
        node_map[e[1].item()].append([idx, 1])
    start_node = -1
    end_node = -1
    # 寻找起始节点和结束节点
    for k, v in node_map.items():
        if len(v) == 1:
            if v[0][1] == 0:
                start_node = k
            if v[0][1] == 1:
                end_node = k
        if start_node != -1 and end_node != -1:
            break
    current_node = start_node
    current_edge = node_map[current_node][0]
    sorted_edge = []
    edge_idx = []
    sorted_edge.append(current_node)
    edge_idx.append(current_edge[0])
    while True:
        next_node = oriented_edge[current_edge[0], (current_edge[1]+1)%2].item()
        sorted_edge.append(next_node)
        if next_node == end_node:
            break
        current_node = next_node
        for e in node_map[current_node]:
            if e[0] != current_edge[0]:
                current_edge = e
                edge_idx.append(current_edge[0])
                break

    return [sorted_edge, edge_idx]

def closed_axis_projection(v):
    # 六个轴方向
    axes = bm.tensor([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=bm.float64)
    dots = bm.matmul(v, axes.T)
    # 取最大点积对应的轴索引作为标签（0~5）
    labels = bm.argmax(dots, dim=1)

    return labels

class PolyCubeProcessor:

    # 六个轴方向
    axes = bm.tensor([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=bm.float64)

    def __init__(self, mesh: TetrahedronMesh):
        self.volume_mesh: TetrahedronMesh = mesh
        self.surface_mesh:TriangleMesh = get_bd_mesh(mesh)
        self.face_area = None
        self.face_center = None
        self.face_normal = self.surface_mesh.cell_normal()
        self.labels = None        # 每个面的轴方向标签（0~5对应±X,±Y,±Z）
        self.charts = []          # 候选图册列表，每个图册包含的面索引
        self.charts_labels = None   # 图册标签，每个图册对应的标签
        self.face2chart = None    # 面到图册的映射
        self.edges = defaultdict(list)  # 候选边 {(face1, face2): [[node_idx], [edge_idx]]}
        self.vertices = defaultdict(int)  # 候选顶点 {vertex_id: count}

    def assign_initial_labels(self):
        """将每个面法向分配到最近的轴方向"""

        # 计算每个法向量与所有轴的点积
        dots = bm.matmul(self.face_normal, self.axes.T)
        # 取最大点积对应的轴索引作为标签（0~5）
        self.labels = bm.argmax(dots, dim=1)

    def build_candidate_charts(self):
        """通过BFS聚类相邻且同标签的面"""
        visited = bm.zeros(self.surface_mesh.number_of_cells(), dtype=bm.bool)
        face2face = self.surface_mesh.cell_to_cell()
        self.face2chart = bm.zeros(self.surface_mesh.number_of_cells(), dtype=bm.int32)
        charts_labels = []
        chat_idx = 0
        for face_id in range(self.surface_mesh.number_of_cells()):
            if not visited[face_id]:
                # BFS遍历邻接面
                queue = [face_id]
                current_label = self.labels[face_id]
                chart = []
                while queue:
                    f = queue.pop(0)
                    if not visited[f] and self.labels[f] == current_label:
                        visited[f] = True
                        chart.append(f)
                        self.face2chart[f] = chat_idx
                        # 获取邻接面
                        neighbors = face2face[f].tolist()
                        queue.extend(neighbors)
                charts_labels.append(current_label)
                self.charts.append(chart)
                chat_idx += 1
        self.charts_labels = bm.tensor(charts_labels, dtype=bm.int32)

    def extract_candidate_edges_vertices(self):
        """提取候选边和顶点"""
        self.edges.clear()
        self.vertices.clear()
        total_edge = self.surface_mesh.edge
        edge2face = self.surface_mesh.edge_to_cell()
        one_by_one_edges = defaultdict(list)
        chart_edges_idx = defaultdict(list)

        # 根据网格边两侧的面所属的 chart 筛选图册边
        is_bd_edge = ((self.face2chart[edge2face[:, 0]] != self.face2chart[edge2face[:, 1]])
                      | (edge2face[:, 0] == edge2face[:, 1]))
        # 遍历边，初步分类候选边界边
        for edge_idx, edge_info in enumerate(edge2face):
            if not is_bd_edge[edge_idx]:
                continue
            left_chart = self.face2chart[edge_info[0]].item()
            right_chart = self.face2chart[edge_info[1]].item()
            # 获取该边起点与终点
            start_idx = total_edge[edge_idx, 0]
            end_idx = total_edge[edge_idx, 1]
            # 统一相同两个图的边界标识
            if left_chart > right_chart:
                left_chart, right_chart = right_chart, left_chart
                start_idx, end_idx = end_idx, start_idx
            edge_key = frozenset([left_chart, right_chart])
            one_by_one_edges[edge_key].append([start_idx, end_idx])
            chart_edges_idx[edge_key].append(edge_idx)
        # 对每条候选边界边的顶点进行排序
        for key in one_by_one_edges.keys():
            sorted_edge = edge_sort(bm.array(one_by_one_edges[key], dtype=bm.int32))
            sorted_edge[1] = bm.array(chart_edges_idx[key], dtype=bm.int32)[sorted_edge[1]].tolist()
            self.edges[key] = sorted_edge
            # 统计候选顶点
            for idx in sorted_edge[0]:
                self.vertices[idx] += 1
        # 过滤掉小于3的顶点
        self.vertices = defaultdict(int, {k: count for k, count in self.vertices.items() if count >= 3})

    def straighten_edges(self, max_iter=5):
        """
        调整边界边周围的面标签以减少锯齿
        Parameters
        ----------
        max_iter

        Returns
        -------

        """
        total_edge = self.surface_mesh.edge
        edge2face = self.surface_mesh.edge_to_cell()
        for j in range(max_iter):
            chart_edge_node_idx = []
            chart_edge_idx = []
            chart_edge_flag = []
            # 获取当前图册的边界边相关信息
            for key, val in self.edges.items():
                # {(face1, face2): [[node_idx], [edge_idx]]}
                # 边界左右两侧图册索引
                chart_edge_flag.append(sorted(key))
                # 边界边的节点索引
                chart_edge_node_idx.append(val[0])
                # 边界包含的网格边的索引
                chart_edge_idx.append(val[1])
            # 遍历所有图册的边界边
            for i, chart_flag in enumerate(chart_edge_flag):
                # 遍历当前图册边包含的所有网格边
                for e, edge in enumerate(chart_edge_idx[i]):
                    if e == (len(chart_edge_idx[i])-1):
                        # 如果是最后一条边，跳过
                        break
                    # 获取当前边的左右面索引，通过网格拓扑信息获取，因此可能与之前图册的边界方向相反
                    current_edge_left_face = edge2face[edge, 0]
                    current_edge_right_face = edge2face[edge, 1]
                    # 获取图册边的起始点
                    n = chart_edge_node_idx[i][e]
                    # 获取网格边的起始点
                    n_o = total_edge[edge, 0]
                    if n != n_o:
                        # 如果网格边与图册边起始点不一致，交换左右面索引
                        current_edge_left_face, current_edge_right_face = current_edge_right_face, current_edge_left_face
                    # 记录左右两侧面的图册索引
                    left_chart_flag = self.face2chart[current_edge_left_face].item()
                    right_chart_flag = self.face2chart[current_edge_right_face].item()
                    # 下条边信息
                    next_edge = chart_edge_idx[i][e+1]
                    next_edge_left_face = edge2face[next_edge, 0]
                    next_edge_right_face = edge2face[next_edge, 1]
                    n = chart_edge_node_idx[i][e+1]
                    n_o = total_edge[next_edge, 0]
                    if n != n_o:
                        # 如果网格边与图册边起始点不一致，交换左右面索引
                        next_edge_left_face, next_edge_right_face = next_edge_right_face, next_edge_left_face
                    if current_edge_left_face == next_edge_left_face:
                        # 左侧出现锯齿
                        # 更新左侧面标签，新标签为右侧面标签
                        self.labels[current_edge_left_face] = self.labels[current_edge_right_face]
                        # 将左侧面合并到右侧面所在的图册中
                        self.face2chart[current_edge_left_face] = right_chart_flag
                        # 更新图册信息
                        self.charts[left_chart_flag].remove(current_edge_left_face)
                        self.charts[right_chart_flag].append(current_edge_left_face)
                        continue
                    if current_edge_right_face == next_edge_right_face:
                        # 右侧出现锯齿
                        # 更新右侧面标签，新标签为左侧面标签
                        self.labels[current_edge_right_face] = self.labels[current_edge_left_face]
                        # 将右侧面合并到左侧面所在的图册中
                        self.face2chart[current_edge_right_face] = left_chart_flag
                        # 更新图册信息
                        self.charts[right_chart_flag].remove(current_edge_right_face)
                        self.charts[left_chart_flag].append(current_edge_right_face)
                        continue
            # 使用新的图册信息，更新候选边和顶点
            self.extract_candidate_edges_vertices()

    def merge_small_charts(self, min_size=5):
        """
        合并小的图册
        Parameters
        ----------
        min_size : int
            最小图册大小

        Returns
        -------

        """
        # TODO: Test, 合并之后更新相关属性
        while True:
            chart_num = len(self.charts)
            # TODO: 考虑改成稀疏矩阵，或者其他更紧凑的数据结构
            chart_adjacency = bm.zeros((chart_num, chart_num), dtype=bm.bool)
            for edge_key in self.edges.keys():
                chart1, chart2 = edge_key
                chart_adjacency[chart1, chart2] = True
                chart_adjacency[chart2, chart1] = True
            chart_neighbors = []
            chart_neighbors_num = bm.sum(chart_adjacency, axis=1)
            chart_face_num = bm.zeros(chart_num, dtype=bm.int32)
            for idx, c in enumerate(self.charts):
                chart_face_num[idx] = len(c)
                neighbors = bm.nonzero(chart_adjacency[idx])[0]
                chart_neighbors.append(neighbors.tolist())
            for idx, c in enumerate(self.charts):
                if (chart_neighbors_num[idx] < 4) or (chart_face_num[idx] < min_size):
                    # 合并到邻接的图册中
                    neighbors = chart_neighbors[idx]
                    max_face_neighbor = neighbors[0]
                    for n in neighbors[1:]:
                        if chart_face_num[n] > chart_face_num[max_face_neighbor]:
                            max_face_neighbor = n
                    self.charts[max_face_neighbor].extend(c)
                    self.charts.pop(idx)
                    for f in c:
                        self.face2chart[f] = max_face_neighbor
                        self.labels[f] = self.charts_labels[max_face_neighbor]
                    # 使用新的图册信息，更新候选边和顶点
                    self.extract_candidate_edges_vertices()
                    continue
            break

    def validate_topology(self):
        """检查PolyCube拓扑有效性"""
        # TODO: Test
        chart_num = len(self.charts)
        chart_adjacency = bm.zeros((chart_num, chart_num), dtype=bm.bool)
        for edge_key in self.edges.keys():
            chart1, chart2 = edge_key
            chart_adjacency[chart1, chart2] = True
            chart_adjacency[chart2, chart1] = True
        chart_neighbors = []
        chart_neighbors_num = bm.sum(chart_adjacency, axis=1)
        chart_face_num = bm.zeros(chart_num, dtype=bm.int32)
        for idx, c in enumerate(self.charts):
            chart_face_num[idx] = len(c)
            neighbors = bm.nonzero(chart_adjacency[idx])
            chart_neighbors.append(neighbors[0].tolist())
        valid = True
        for idx, chart in enumerate(self.charts):
            if (chart_neighbors_num[idx] < 4):
                print(f"chart {idx} has less than 4 neighbors")
                valid = False
                break
            current_chart_label = self.charts_labels[idx]
            current_chart_normal = self.axes[current_chart_label]
            for n in chart_neighbors[idx]:
                neighbor_normal = self.axes[self.charts_labels[n]]
                if bm.dot(current_chart_normal, neighbor_normal) < 0:
                    print(f"chart {idx} and chart {n} have opposite normals")
                    valid = False
                    break
        for k, v in self.vertices.items():
            if v != 3:
                print(f"vertex {k} has {v} edges")
                valid = False
                break

        return valid

    def laplacian_smooth(self, alpha=0.3, max_iter=5):
        """
        拉普拉斯平滑
        Parameters
        ----------
        alpha : float
            平滑系数
        max_iter : int
            最大迭代次数

        Returns
        -------

        """
        origin_node = self.surface_mesh.node
        for i in range(max_iter):
            for e in self.edges.values():
                node_id = bm.array(e[0], dtype=bm.int32)
                edge_node = origin_node[node_id]
                for n in range(1, len(edge_node)-1):
                    # 计算拉普拉斯平滑
                    edge_node[n] = (1-alpha)*edge_node[n] + alpha*(edge_node[n-1] + edge_node[n+1]) / 2
                # edge_node[0] = (1 - alpha) * edge_node[0] + alpha * edge_node[1]
                # edge_node[-1] = (1 - alpha) * edge_node[-1] + alpha * edge_node[-2]
                origin_node[node_id] = edge_node
        self.surface_mesh.node = origin_node

    def edge_projection(self):
        # 将边界边投影到最近的轴方向
        origin_node = self.surface_mesh.node
        for key, val in self.edges.items():
            left_chart, right_chart = sorted(key)
            node_id = bm.array(val[0], dtype=bm.int32)
            edge_node = origin_node[node_id]
            first_node = bm.copy(edge_node[0])
            left_normal = self.axes[self.charts_labels[left_chart]]
            right_normal = self.axes[self.charts_labels[right_chart]]
            normal_direction = bm.cross(left_normal, right_normal)
            new_node = first_node + bm.einsum('nv, v, d->nd', edge_node-first_node, normal_direction, normal_direction)
            origin_node[node_id] = new_node
        self.surface_mesh.node = origin_node

    def detect_turning_points(self):
        origin_node = self.surface_mesh.node
        turning_points = []
        for key, val in self.edges.items():
            left_chart, right_chart = sorted(key)
            node_id = bm.array(val[0], dtype=bm.int32)
            edge_node = origin_node[node_id]
            first_node = bm.copy(edge_node[0])
            left_normal = self.axes[self.charts_labels[left_chart]]
            right_normal = self.axes[self.charts_labels[right_chart]]
            normal_direction = bm.cross(left_normal, right_normal)
            t = bm.einsum('nd, d->n', edge_node - first_node, normal_direction)
            t0 = t[0]
            for n in range(1, len(t)):
                t_now = t[n]
                if t_now < t0:
                    # 反向
                    turning_points.append(val[0][n])
                t0 = t_now
        return turning_points



if __name__ == '__main__':
    volume_mesh = TetrahedronMesh.from_box(nx=3, ny=3, nz=3)
    cell = volume_mesh.cell
    node = volume_mesh.node
    node[39, 0] = 2/ 3
    node[39, 1] = 1 / 3
    node[39, 2] = 7/6

    node[35, 0] = 8/12
    node[35, 1] = 0
    node[35, 2] = 1

    node[55, 0] = 1
    node[55, 1] = 3/12
    node[55, 2] = 5/6

    node[59, 0] = 1
    node[59, 1] = 2/3
    node[59, 2] = 5/6
    volume_mesh = TetrahedronMesh(node, cell)
    # volume_mesh.to_vtk(fname='volume_mesh.vtu')
    # volume_mesh = pickle.load(open("optimized_mesh.pkl", "rb"))
    # volume_mesh.to_vtk(fname='volume_mesh_origin.vtu')

    bd_mesh = get_bd_mesh(volume_mesh)
    # bd_mesh.to_vtk(fname='bd_mesh.vtu')
    processor = PolyCubeProcessor(volume_mesh)
    processor.assign_initial_labels()
    bd_mesh.celldata['initial_labels'] = bm.copy(processor.labels)
    processor.build_candidate_charts()
    bd_mesh.celldata['charts'] = bm.copy(processor.face2chart)
    # bd_mesh.to_vtk(fname='bd_mesh.vtu')

    processor.extract_candidate_edges_vertices()
    # 提取图册边界网格
    edge = bd_mesh.edge
    node = bd_mesh.node
    edge_mesh = IntervalMesh(node, edge)

    edge_date = bm.zeros(len(edge), dtype=bm.float64)
    node_date = bm.zeros(len(node), dtype=bm.float64)
    for i, v in enumerate(processor.edges.items()):
        for e in v[1][1]:
            edge_date[e] = (i + 1)*100
    for i, v in enumerate(processor.vertices.items()):
        node_date[v[0]] = (i + 1)*100
    edge_mesh.celldata['edge_date'] = edge_date
    edge_mesh.nodedata['node_date'] = node_date

    # 边拉直
    processor.straighten_edges(max_iter=5)

    edge_date_after = bm.zeros(len(edge), dtype=bm.float64)
    node_date_after = bm.zeros(len(node), dtype=bm.float64)
    for i, v in enumerate(processor.edges.items()):
        for e in v[1][1]:
            edge_date_after[e] = (i + 1) * 100
    for i, v in enumerate(processor.vertices.items()):
        node_date_after[v[0]] = (i + 1) * 100
    edge_mesh.celldata['edge_date_after'] = edge_date_after
    edge_mesh.nodedata['node_date_after'] = node_date_after
    # edge_mesh.to_vtk(fname='edge_mesh.vtu')
    bd_mesh.celldata['charts_after'] = processor.face2chart
    # bd_mesh.to_vtk(fname='bd_mesh.vtu')


    processor.merge_small_charts(min_size=5)
    is_valid = processor.validate_topology()
    print(f"Topology is valid: {is_valid}")

    processor.laplacian_smooth(alpha=0.3, max_iter=5)
    # bd_mesh_smooth = processor.surface_mesh
    # bd_mesh_smooth.to_vtk(fname='bd_mesh_smooth.vtu')

    processor.edge_projection()
    bd_mesh_project = processor.surface_mesh
    bd_mesh_project.celldata['charts'] = bm.copy(processor.face2chart)
    bd_mesh_project.to_vtk(fname='bd_mesh_project.vtu')

    turning_points = processor.detect_turning_points()
    print("turning points:", turning_points)


    print("candidate charts:")
    for idx, chart in enumerate(processor.charts):
        print(chart)
        print(processor.axes[processor.charts_labels[idx]])
    print("candidate edges:")
    for key in processor.edges.keys():
        print(key, processor.edges[key])
    print("candidate vertices:")
    for key in processor.vertices.keys():
        print(key, processor.vertices[key])

















