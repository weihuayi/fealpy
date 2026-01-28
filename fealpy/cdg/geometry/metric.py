from fealpy.backend import bm
from fealpy.mesh import TriangleMesh


class MeshMetric:
    """
    离散黎曼度量计算类。

    职责：
    1. 计算网格的边长、面积、角度。
    2. 计算离散高斯曲率。
    3. 支持基于自定义边长进行计算 (为 Ricci Flow 预留)。
    """

    def __init__(self, mesh: TriangleMesh):
        self.mesh = mesh

    def edge_length(self, point=None):
        """
        计算所有边的长度。

        Args:
            point (bm.ndarray, optional): 如果提供了顶点坐标，则基于该坐标计算。
                                          否则使用 mesh.node。
        Returns:
            l (bm.ndarray): 形状为 (NE, ) 的数组，对应 mesh.ds.edge 的长度。
        """
        if point is None:
            point = self.mesh.entity('node')

        edges = self.mesh.edge
        v0 = point[edges[:, 0]]
        v1 = point[edges[:, 1]]

        return bm.linalg.norm(v0 - v1, axis=1)

    def cell_edge_lengths(self, l: bm.ndarray = None):
        """
        获取每个三角形三条边的长度。
        这是计算角度和面积的基础。

        Args:
            l (bm.ndarray, optional): 预计算好的边长数组 (NE, )。
                                      如果为空，则根据当前坐标计算。
                                      *Ricci流中会传入更新后的 l*

        Returns:
            l_sq (bm.ndarray): 形状 (NC, 3)。
                               每行对应一个单元，三列对应三条边 [e0, e1, e2] 的长度。
                               顺序通常对应于局部顶点索引 [v1-v2, v2-v0, v0-v1]。
        """
        if l is None:
            l = self.edge_length()

        cell2edge = self.mesh.cell2edge
        # 获取每个单元三条边的长度
        # 注意：Fealpy 的 cell2edge 返回的顺序对应于局部顶点的对边
        return l[cell2edge]

    def corner_angles(self, l: bm.ndarray = None):
        """
        计算每个三角形的三个内角（使用余弦定理）。

        Args:
            l (bm.ndarray, optional): 边长数组。

        Returns:
            angles (bm.ndarray): 形状 (NC, 3)。对应每个单元的三个内角。
        """
        # 1. 获取三角形三边长
        # l_cell 的列通常对应: 
        # col 0: 对面顶点 0 的边 (v1-v2)
        # col 1: 对面顶点 1 的边 (v2-v0)
        # col 2: 对面顶点 2 的边 (v0-v1)
        edges = self.cell_edge_lengths(l)

        l0, l1, l2 = edges[:, 0], edges[:, 1], edges[:, 2]

        # 2. 余弦定理
        # cos(theta_0) = (l1^2 + l2^2 - l0^2) / (2 * l1 * l2)
        # 为了数值稳定性，clip 到 [-1, 1] 之间
        cos_0 = bm.clip((l1 ** 2 + l2 ** 2 - l0 ** 2) / (2 * l1 * l2), -1.0, 1.0)
        cos_1 = bm.clip((l0 ** 2 + l2 ** 2 - l1 ** 2) / (2 * l0 * l2), -1.0, 1.0)
        cos_2 = bm.clip((l0 ** 2 + l1 ** 2 - l2 ** 2) / (2 * l0 * l1), -1.0, 1.0)

        angles = bm.column_stack([
            bm.arccos(cos_0),
            bm.arccos(cos_1),
            bm.arccos(cos_2)
        ])

        return angles

    def cell_area(self, l: bm.ndarray = None):
        """
        使用海伦公式 (Heron's Formula) 计算三角形面积。
        (这也支持 Ricci 流中边长改变后的面积计算)
        """
        edges = self.cell_edge_lengths(l)
        a, b, c = edges[:, 0], edges[:, 1], edges[:, 2]

        # 半周长
        s = (a + b + c) / 2.0

        # 海伦公式 area = sqrt(s * (s-a) * (s-b) * (s-c))
        # 加上 max(..., 1e-12) 防止数值误差导致负数开根号
        val = s * (s - a) * (s - b) * (s - c)
        return bm.sqrt(bm.maximum(val, 1e-12))

    def gaussian_curvature(self, l: bm.ndarray = None):
        """
        计算离散高斯曲率 (Discrete Gaussian Curvature)。

        公式:
            内部点: K = 2*pi - sum(theta)
            边界点: K = pi - sum(theta) (或者是 target curvature)

        Returns:
            K (bm.ndarray): 形状 (NV, )，每个顶点的曲率。
        """
        NC = self.mesh.number_of_cells()
        NV = self.mesh.number_of_nodes()
        cell = self.mesh.entity('cell')

        # 1. 计算所有角
        angles = self.corner_angles(l)  # (NC, 3)

        # 2. 将角度累加到对应的顶点上
        # angle_sum[i] 表示第 i 个顶点周围的角度和
        angle_sum = bm.zeros(NV)
        bm.add.at(angle_sum, cell, angles)

        # 3. 计算曲率
        # 默认目标值为 2*pi
        target = bm.full(NV, 2 * bm.pi)

        # 处理边界：如果是边界点，目标角和应该是 pi
        is_bd = self.mesh.ds.boundary_node_flag()
        target[is_bd] = bm.pi

        K = target - angle_sum
        return K

    def cotangent_weights(self, l: bm.ndarray = None):
        """
        计算余切权重 (Cotangent Weights)。
        这是组装拉普拉斯矩阵的核心。

        Returns:
            cot_weights (bm.ndarray): 形状 (NC, 3)。
                                      对应每个角的 cot 值，顺序与 cell 顶点对应。
        """
        angles = self.corner_angles(l)
        # cot(x) = 1 / tan(x) = cos(x) / sin(x)
        # 加上 epsilon 防止除以 0
        sin_a = bm.sin(angles)
        cos_a = bm.cos(angles)

        cot_a = cos_a / (sin_a + 1e-12)
        return cot_a