from abc import ABC, abstractmethod
import numpy as np

from fealpy.geometry.utils import delta_angle_calculator
from numpy import sin, cos, tan, pi, arctan, arctan2, radians, sqrt

from scipy.optimize import fsolve
from fealpy.mesh.quadrangle_mesh import QuadrangleMesh
from fealpy.geometry.utils import *
from app.gearx.utils import *

class Gear(ABC):
    def __init__(self, m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, nw, tooth_width, material=None, rotation_direction=1, center=(0,0,0), name=None):
        """

        @param m_n: 法向模数
        @param z: 齿数
        @param alpha_n: 法向压力角
        @param beta: 螺旋角
        @param x_n: 法向变位系数
        @param hac: 齿顶高系数
        @param cc: 顶隙系数
        @param rcc: 刀尖圆弧半径系数
        @param jn: 法向侧隙
        @param n1: 渐开线分段数
        @param n2: 过渡曲线分段数
        @param n3: 齿轮内部分段数
        @param na: 齿顶分段数
        @param nf: 齿根圆部分分段数（一侧，非最大圆角时）
        @param nw: 沿齿宽分段数
        @param tooth_width: 齿宽
        @param material: 齿轮材料
        @param rotation_direction: 旋转方向，1 为右旋齿轮，-1 为左旋齿轮，默认为右旋
        @param center: 齿轮中心坐标，默认为原点
        @param name: 齿轮名称
        """
        # 参数类型检查
        for param in [z, n1, n2, n3, na, nf, nw]:
            if not isinstance(param, int) or (isinstance(param, float) and not param.is_integer()):
                raise TypeError(
                    f'The provided value {param} is not an integer or cannot be safely converted to an integer.')
        assert rotation_direction in [-1, 1, -1., 1., "-1", "1"], 'The rotation direction must be either 1 or -1.'

        self.m_n = m_n
        self.z = z
        self.alpha_n = alpha_n if alpha_n < 2 * pi else radians(alpha_n)
        self.beta = beta if beta < 2 * pi else radians(beta)
        self.x_n = x_n
        self.hac = hac
        self.cc = cc
        self.rcc = rcc
        self.jn = jn
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.na = na
        self.nf = nf
        self.nw = nw
        self.tooth_width = tooth_width
        self.mesh = None
        self.hex_mesh = None
        self._material = material
        self.rotation_direction = rotation_direction
        self.center = center
        self.name = name

        # 端面变位系数
        self.x_t = self.x_n * cos(self.beta)
        # 端面压力角
        self.alpha_t = arctan(tan(self.alpha_n) / cos(self.beta))
        # 端面模数
        self.m_t = self.m_n / cos(self.beta)
        # 分度圆直径与半径
        self.d = self.m_t * self.z
        self.r = self.d / 2
        # 基圆（base circle）直径与半径
        self.d_b = self.d * cos(self.alpha_t)
        self.r_b = self.d_b / 2

    @abstractmethod
    def get_involute_points(self):
        pass

    @abstractmethod
    def get_tip_intersection_points(self):
        pass

    @abstractmethod
    def get_transition_points(self):
        pass

    @abstractmethod
    def get_profile_points(self):
        pass

    @abstractmethod
    def generate_mesh(self):
        pass

    def show_mesh(self, save_path=None):
        if self.mesh is None:
            raise AssertionError('The mesh is not yet created.')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(38, 20))
        self.mesh.add_plot(ax, linewidth=0.1)
        if save_path is not None:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

    def cylindrical_to_cartesian(self, d, width):
        """
        给定宽度坐标以及半径，计算其对应的笛卡尔坐标，适用于外齿轮，内齿轮需要进一步测试
        TODO: 内齿轮的计算，实现自定义所在齿
        :param d: 节点所在圆弧的直径
        :param width: 宽度
        :return: 节点笛卡尔坐标
        """
        r = d / 2

        if isinstance(r, (float, int)):
            def involutecross(t2):
                return self.get_tip_intersection_points(t2) - r

            # 计算端面点（z=0）坐标
            t = fsolve(involutecross, self.m_n)[0]
            point_t = np.zeros(3)
            point_t[0:2] = self.get_involute_points(t)

            # 根据螺旋线与当前宽度，计算实际坐标
            total_width = self.tooth_width
            t2 = width / total_width
            point = get_helix_points(point_t, self.beta, self.r, total_width, t2, self.rotation_direction)
        elif isinstance(r, (np.ndarray, list)):
            point_t = np.zeros((len(r), 3))
            total_width = self.tooth_width
            t2 = width / total_width
            point = np.zeros((len(r), 3))
            for i in range(len(r)):
                def involutecross(t2):
                    return self.get_tip_intersection_points(t2) - r[i]

                # 计算端面点（z=0）坐标
                t = fsolve(involutecross, self.m_n)[0]
                point_t[i, 0:2] = self.get_involute_points(t)
                point[i] = get_helix_points(point_t[i], self.beta, self.r, total_width, t2[i], self.rotation_direction)
        point[..., 0:2] = np.dot(point[..., 0:2], np.array([[0, -1], [1, 0]]))
        return point

    def generate_hexahedron_mesh(self):
        """
        根据齿轮端面网格，使用扫掠法，生成整体网格

        :return: 端面四边形网格对应的六面体网格
        """
        quad_mesh = self.mesh if self.mesh is not None else self.generate_mesh()
        node = quad_mesh.node
        cell = quad_mesh.cell
        beta = self.beta
        r = self.r
        tooth_width = self.tooth_width
        nw = self.nw
        rotation_direction = self.rotation_direction
        # 数据处理，将二维点转换为三维点
        new_node = np.zeros((len(node), 3))
        new_node[:, 0:2] = node
        one_section_node_num = len(new_node)

        # 创建齿轮整体网格
        # 拉伸节点
        volume_node = sweep_points(new_node, beta, r, tooth_width, nw, rotation_direction).reshape(-1, 3)
        # 将端面四边形单元拉伸为六面体单元
        volume_cell = np.zeros((nw, len(cell), 8), dtype=np.int64)
        cell_domain_tag = np.zeros((nw, len(cell)))
        cell_tooth_tag = np.zeros((nw, len(cell)))
        transverse_cell_domain_tag = quad_mesh.celldata['cell_domain_tag'][None, :]
        transverse_cell_tooth_tag = quad_mesh.celldata['cell_tooth_tag'][None, :]
        # 填充单元的节点索引
        for i in range(nw):
            volume_cell[i, :, 0:4] = cell + i * one_section_node_num
            volume_cell[i, :, 4:8] = cell + (i + 1) * one_section_node_num
            cell_domain_tag[i, :] = transverse_cell_domain_tag
            cell_tooth_tag[i, :] = transverse_cell_tooth_tag
        volume_cell = volume_cell.reshape(-1, 8)
        cell_domain_tag = cell_domain_tag.reshape(-1)
        cell_tooth_tag = cell_tooth_tag.reshape(-1)

        hex_mesh = HexahedronMesh(volume_node, volume_cell)
        hex_mesh.celldata['cell_domain_tag'] = cell_domain_tag
        hex_mesh.celldata['cell_tooth_tag'] = cell_tooth_tag
        self.hex_mesh = hex_mesh

        return hex_mesh

    def find_node_location_kd_tree(self, target_node, error=1e-3):
        """
        查找目标节点在六面体网格中的位置，基于 kd_tree
        :param target_node: 目标节点坐标
        :param error: 误差限制
        :return: 目标节点所在的单元索引，节点所在单元局部面索引，节点关于所在节点参数，若未找到则返回-1
        """
        # 使用 kd_tree 算法，先计算所有单元的重心坐标，再根据重心坐标与target_node构建 kd_tree
        if not hasattr(self, 'hex_mesh') or self.hex_mesh is None:
            raise ValueError('The hex_mesh attribute is not set.')
        mesh = self.hex_mesh
        cell_barycenter = mesh.entity_barycenter('cell')
        # 计算每个单元重心坐标与 target_node 的距离，并排序从而构建kd_tree
        distance = np.linalg.norm(cell_barycenter - target_node, axis=1)
        kd_tree = np.argsort(distance)
        # 获取网格实体信息
        cell = mesh.cell
        node = mesh.node
        local_tetra = np.array([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=np.int32)
        tetra_local_face = np.array([
            (1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)], dtype=np.int32)
        tetra_face_to_hex_face = np.array([
            (3, -1, -1, 0),
            (3, -1, -1, 4),
            (1, -1, -1, 4),
            (1, -1, -1, 2),
            (5, -1, -1, 2),
            (5, -1, -1, 0)], dtype=np.int32)
        # 根据网格单元测度设置误差限制
        error = np.max(mesh.entity_measure('cell')) * error
        # 遍历单元搜寻
        for i in range(len(kd_tree)):
            cell_idx = kd_tree[i]
            cell_node = node[cell[cell_idx]]
            # 将一个六面体分成六个四面体，计算目标点是否在某一个四面体内（包括边界面与点）
            # 若六个四面体中有一个包含目标点，则返回当前六面体单元索引
            tetras = cell_node[local_tetra]
            # 遍历六个四面体
            for j, tetra in enumerate(tetras):
                for i in range(tetra_local_face.shape[0]):
                    current_face_node = tetra[tetra_local_face[i]]
                    v = -sign_of_tetrahedron_volume(current_face_node[0], current_face_node[1], current_face_node[2],
                                                    target_node)
                    if v < 0 and abs(v - 0) > error:
                        break
                    if (v > 0 or abs(v - 0) < error) and i == tetra_local_face.shape[0] - 1:
                        t = (target_node[2] - cell_node[0, 2]) / (cell_node[4, 2] - cell_node[0, 2]);
                        r_points = np.sqrt(np.sum(cell_node[0:4, 0:2] ** 2, axis=-1))
                        tooth_helix = (cell_node[4, 2] - cell_node[0, 2]) * tan(self.beta) / self.r
                        start_angle = arctan2(cell_node[0:4, 1], cell_node[0:4, 0])
                        # 构建目标节点所在截面四边形
                        t_node = np.zeros((4, 2))
                        t_node[:, 0] = r_points * cos((tooth_helix * t) + start_angle)
                        t_node[:, 1] = r_points * sin((tooth_helix * t) + start_angle)
                        P00 = t_node[0]
                        P10 = t_node[1]
                        P11 = t_node[2]
                        P01 = t_node[3]
                        P = target_node[0:2]
                        # 计算二元一次方程组系数
                        # (P00-P10)x(P01-P11)u**2 + ((P-P00)x(P01-P11)-(P-P01)x(P00-P10))u + (P-P00)x(P-P01) = 0
                        a = (P00[0] * P01[1] - P01[0] * P00[1] - P00[0] * P11[1] + P01[0] * P10[1]
                             - P10[0] * P01[1] + P11[0] * P00[1] + P10[0] * P11[1] - P11[0] * P10[1])
                        b = (P[0] * P01[1] - P[0] * P00[1] + P00[0] * P[1] - P01[0] * P[1]
                             + P[0] * P10[1] - P[0] * P11[1] + P11[0] * P[1] - P10[0] * P[1]
                             - 2 * P00[0] * P01[1] + 2 * P01[0] * P00[1] +
                             P00[0] * P11[1] - P01[0] * P10[1] + P10[0] * P01[1] - P11[0] * P00[1])
                        c = (P[0] * P00[1] - P[0] * P01[1] + P01[0] * P[1] - P00[0] * P[1]
                             + P00[0] * P01[1] - P01[0] * P00[1])

                        delta = b ** 2 - 4 * a * c
                        u = -1
                        if delta < 0:
                            break
                        else:
                            u0 = (-b + np.sqrt(delta)) / (2 * a)
                            u1 = (-b - np.sqrt(delta)) / (2 * a)
                            if 0 - error * 10 <= u0 <= 1 + error * 10:
                                u = u0
                            elif 0 - error * 10 <= u1 <= 1 + error * 10:
                                u = u1
                        v = -1
                        if u != -1:
                            Pu0 = (1 - u) * P00 + u * P10
                            Pu1 = (1 - u) * P01 + u * P11
                            v0 = (P[0] - Pu0[0]) / (Pu1[0] - Pu0[0])
                            v1 = (P[1] - Pu0[1]) / (Pu1[1] - Pu0[1])
                            if abs(v0 - v1) < error ** 2:
                                if 0 - error <= v0 <= 1 + error:
                                    v = v0
                        w = t
                        return cell_idx, tetra_face_to_hex_face[j, i], (u, v, w)
        raise ValueError('Target node not found in any cell.')

    def get_profile_node_index(self, tooth_tag=None):
        """
        寻找目标齿两侧齿廓上的节点索引及坐标
        :param tooth_tag: 目标齿编号，默认为 None，即所有齿
        :return: 齿廓节点索引及坐标
        """
        if not hasattr(self, 'hex_mesh') or self.hex_mesh is None:
            raise ValueError('The hex_mesh attribute is not set.')
        # tooth_tag 输入类型检测，只能是整数或整数列表，并将其转换为列表
        if tooth_tag is not None:
            if isinstance(tooth_tag, int):
                tooth_tag = [tooth_tag]
            elif isinstance(tooth_tag, (list, tuple)):
                tooth_tag = list(tooth_tag)
            else:
                raise TypeError('The tooth_tag must be an integer or a list of integers.')
        quad_mesh = self.mesh
        hex_mesh = self.hex_mesh
        t_NN = quad_mesh.number_of_nodes()
        n1 = self.n1
        nw = self.nw
        hex_node = hex_mesh.node

        is_bd_cell = quad_mesh.boundary_cell_flag()
        domain_flag_left = quad_mesh.celldata["cell_domain_tag"] == 6
        domain_flag_right = quad_mesh.celldata["cell_domain_tag"] == 5
        if tooth_tag is None:
            folder = self.z
            cell_flag_left = is_bd_cell & domain_flag_left
            cell_flag_right = is_bd_cell & domain_flag_right
        else:
            folder = len(tooth_tag)
            tooth_flag = np.isin(quad_mesh.celldata["cell_tooth_tag"], tooth_tag)

            cell_flag_left = is_bd_cell & domain_flag_left & tooth_flag
            cell_flag_right = is_bd_cell & domain_flag_right & tooth_flag

        # 右侧齿面
        # TODO: 考虑是否给单元局部起始节点编号加检测，即不一定从 0 和 2 开始
        cell_idx_right = np.where(cell_flag_right)[0].reshape(folder, -1)[..., 0:n1]
        tooth_profile_cell_right = quad_mesh.cell[cell_idx_right]
        tooth_profile_node_right = np.zeros((folder, nw + 1, n1 + 1), dtype=np.int32)
        tooth_profile_node_right[..., 0, 0:n1] = tooth_profile_cell_right[..., :, 0]
        tooth_profile_node_right[..., 0, -1] = tooth_profile_cell_right[..., -1, 1]

        # 左侧齿面
        cell_idx_left = np.flip(np.flip(np.where(cell_flag_left)[0]).reshape(folder, -1)[..., 0:n1], axis=0)
        tooth_profile_cell_left = quad_mesh.cell[cell_idx_left]
        tooth_profile_node_left = np.zeros((folder, nw + 1, n1 + 1), dtype=np.int32)
        tooth_profile_node_left[..., 0, 0:n1] = tooth_profile_cell_left[..., :, 2]
        tooth_profile_node_left[..., 0, -1] = tooth_profile_cell_left[..., -1, 1]

        for i in range(1, nw+1):
            tooth_profile_node_right[..., i, :] = tooth_profile_node_right[..., 0, :] + i * t_NN
            tooth_profile_node_left[..., i, :] = tooth_profile_node_left[..., 0, :] + i * t_NN

        return (tooth_profile_node_right, tooth_profile_node_left), (hex_node[tooth_profile_node_right], hex_node[tooth_profile_node_left])

    @property
    def material(self):
        return self._material

    @material.setter
    def value(self, new_material):
        self._material = new_material
        pass


class ExternalGear(Gear):
    def __init__(self, m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, nw, chamfer_dia, inner_diam, tooth_width,
                 material=None, rotation_direction=1, center=(0,0,0), name=None):
        """

        @param m_n: 法向模数
        @param z: 齿数
        @param alpha_n: 法向压力角
        @param beta: 螺旋角
        @param x_n: 法向变位系数
        @param hac: 齿顶高系数
        @param cc: 顶隙系数
        @param rcc: 刀尖圆弧半径系数
        @param jn: 法向侧隙
        @param n1: 渐开线分段数
        @param n2: 过渡曲线分段数
        @param n3: 齿轮内部分段数
        @param na: 齿顶分段数
        @param nf: 齿根圆部分分段数（一侧，非最大圆角时）
        @param nw: 沿齿宽分段数
        @param chamfer_dia: 倒角高度（直径方向）
        @param inner_diam: 轮缘内径
        @param tooth_width: 齿宽
        @param material: 齿轮材料
        @param rotation_direction: 旋转方向，1 为右旋齿轮，-1 为左旋齿轮，默认为右旋
        @param center: 齿轮中心坐标，默认为原点
        @param name: 齿轮名称
        """
        super().__init__(m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, nw, tooth_width, material, rotation_direction, center, name)
        self.inner_diam = inner_diam
        self.chamfer_dia = chamfer_dia
        # 齿顶圆直径与半径
        ha = self.m_n * (self.hac + self.x_n)  # 齿顶高
        self.d_a = self.d + 2 * ha
        self.r_a = self.d_a / 2
        # 齿根圆直径与半径
        hf = self.m_n * (self.hac + self.cc - self.x_n)
        self.d_f = self.d - 2 * hf
        self.r_f = self.d_f / 2
        # 有效齿顶圆
        self.effective_da = self.d_a - self.chamfer_dia
        self.effective_ra = self.effective_da / 2
        # 刀具齿顶高与刀尖圆弧半径
        self.ha_cutter = (self.hac + self.cc) * self.m_n
        self.rc = self.m_n * self.rcc

    def get_involute_points(self, t):
        m_n = self.m_n
        alpha_t = self.alpha_t
        beta = self.beta
        r = self.r
        x_t = self.x_t

        k = -(np.pi * m_n / 4 + m_n * x_t * np.tan(alpha_t))
        phi = (t * np.cos(np.pi / 2 - alpha_t) ** 2 + k * np.cos(np.pi / 2 - alpha_t) + t * np.cos(beta) ** 2 * np.sin(
            np.pi / 2 - alpha_t) ** 2) / (r * np.cos(beta) * np.cos(np.pi / 2 - alpha_t))

        xt = (r * np.sin(phi) - phi * r * np.cos(phi) +
              t * np.sin(phi) * np.sin(np.pi / 2 - alpha_t) +
              (np.cos(phi) * (k + t * np.cos(np.pi / 2 - alpha_t))) / np.cos(beta)).reshape(-1, 1)

        yt = (r * np.cos(phi) + phi * r * np.sin(phi) +
              t * np.cos(phi) * np.sin(np.pi / 2 - alpha_t) -
              (np.sin(phi) * (k + t * np.cos(np.pi / 2 - alpha_t))) / np.cos(beta)).reshape(-1, 1)

        points = np.concatenate([xt, yt], axis=-1)
        return points

    def get_tip_intersection_points(self, t):

        points = self.get_involute_points(t)
        return np.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2)

    def get_transition_points(self, t):
        r = self.r
        rc = self.rc  # 刀尖圆弧半径
        ha_cutter = self.ha_cutter  # 刀具齿顶高
        alpha_t = self.alpha_t
        beta = self.beta

        # 刀尖圆弧 y 坐标
        x0 = -np.pi * self.m_n / 2 + (np.pi * self.m_n / 4 - ha_cutter * np.tan(alpha_t) - rc * np.tan(
            0.25 * np.pi - 0.5 * alpha_t))
        # 刀尖圆弧 y 坐标
        y0 = -(ha_cutter - rc) + self.m_n * self.x_t

        phi = (x0 * np.sin(t) + rc * np.cos(t) * np.sin(t) - y0 * np.cos(beta) ** 2 * np.cos(
            t) - rc * np.cos(
            beta) ** 2 * np.cos(t) * np.sin(t)) / (r * np.cos(beta) * np.sin(t))

        xt = (r * np.sin(phi) + np.sin(phi) * (y0 + rc * np.sin(t)) - phi * r * np.cos(phi) +
              (np.cos(phi) * (x0 + rc * np.cos(t))) / np.cos(beta)).reshape(-1, 1)

        yt = (r * np.cos(phi) + np.cos(phi) * (y0 + rc * np.sin(t)) + phi * r * np.sin(phi) -
              (np.sin(phi) * (x0 + rc * np.cos(t))) / np.cos(beta)).reshape(-1, 1)

        points = np.concatenate([xt, yt], axis=-1)
        return points

    def get_transition_intersection_points(self, t):
        points = self.get_transition_points(t)
        return np.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2)

    def get_profile_points(self):
        n1 = self.n1
        n2 = self.n2
        mn = self.m_n
        alpha_t = self.alpha_t
        beta = self.beta
        z = self.z
        x = self.x_t
        effective_da = self.effective_da
        ha_cutter = self.ha_cutter  # 刀具齿顶高
        rc = self.rc  # 刀尖圆弧半径

        points = np.zeros(((n1 + n2 + 1) * 2, 3))

        def involutecross_rb_i(t):
            return self.get_tip_intersection_points(t) - self.r_b

        # t_temp = (mn * x - (ha_cutter - rc + rc * sin(alpha_t))) / cos(alpha_t)
        t1 = fsolve(involutecross_rb_i, mn)[0]

        def involutecross(t2):
            return self.get_tip_intersection_points(t2) - (0.5 * effective_da)

        t2 = fsolve(involutecross, mn)[0]  # 求解渐开线与齿顶圆的交点

        # t3 = 2 * np.pi - alpha_t
        # t4 = 1.5 * np.pi
        def involutecross_rb(t):
            return self.get_transition_intersection_points(t) - self.r_b
        def involutecross_rf(t):
            return self.get_transition_intersection_points(t) - self.r_f

        t3 = fsolve(involutecross_rb, 2*np.pi-alpha_t)[0]
        t4 = fsolve(involutecross_rf, 1.5*np.pi)[0]
        width2 = t3 - t4
        t = np.linspace(t4, t3, n2 + 1)
        points[0:n2 + 1, 0:-1] = self.get_transition_points(t)

        width1 = t2 - t1
        t = np.linspace(t1 + width1 / n1, t2, n1)
        points[n2 + 1:n2 + n1 + 1, 0:-1] = self.get_involute_points(t)

        # 构建对称点
        points[n2 + n1 + 1:, 0] = -points[0:n2 + n1 + 1, 0]
        points[n2 + n1 + 1:, 1] = points[0:n2 + n1 + 1, 1]

        return points

    def generate_mesh(self):
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3
        nf = self.nf
        na = self.na
        rf = self.r_f
        ra = self.effective_ra
        z = self.z
        # 获取齿廓与过渡曲线点列
        points = self.get_profile_points()
        r_inner = self.inner_diam / 2

        one_tooth_angle = abs(delta_angle_calculator(points[0, :2], points[n1 + n2 + 1, :2], input_type="vector"))
        if one_tooth_angle*z > 2*pi:
            max_angle_flag = True
        else:
            max_angle_flag = False
        # TODO: 改用齿根圆角是否超过最大圆角进行判断与分类
        # 两侧过渡曲线之间相连，齿槽底面为一条直线，宽度为 0
        if max_angle_flag:  # 构造关键点
            kp2 = points[n1 + n2 + 1, :2]
            angle_kp2 = np.arctan2(kp2[1], kp2[0])
            delta_angle_kp2 = pi / 2 - angle_kp2
            angle_kp1 = pi / 2 - delta_angle_kp2
            angle_kp4 = pi / 2 + delta_angle_kp2
            kp5 = points[0, :2]
            kp6 = points[n1 + 2 * n2 + 1, :2]
            kp8 = points[n2, :2]
            kp7 = points[-1, :2]
            kp9 = points[n1 + n2, :2]

            angle_kp7 = np.arctan2(kp7[1], kp7[0])
            delta_angle_kp7 = pi / 2 - angle_kp7
            angle_kp15 = pi / 2 - delta_angle_kp7 / 2
            angle_kp17 = pi / 2 + delta_angle_kp7 / 2
            kp15 = np.array([ra * np.cos(angle_kp15), ra * np.sin(angle_kp15)])
            kp17 = np.array([ra * np.cos(angle_kp17), ra * np.sin(angle_kp17)])

            kp0 = np.array([r_inner * np.cos(angle_kp2), r_inner * np.sin(angle_kp2)])
            kp3 = np.array([r_inner * np.cos(angle_kp4), r_inner * np.sin(angle_kp4)])

            kp10 = np.array([0, r_inner])
            kp13 = np.array([0, ra])

            # 单侧弧长与点数，计算中轴上点参数
            distance = np.sqrt(np.sum(np.diff(points[:n1 + n2 + 1, :2], axis=0) ** 2, axis=1))
            length2 = np.sum(distance[:n1])
            length3 = np.sum(distance[n1:n1 + n2])
            length1 = np.sqrt(np.sum((kp5 - kp3) ** 2))

            n_total = n1 * 0.618 + n2 * 0.618 + n3
            length2_n = length2 * (n1 * 0.618 / n_total)
            length3_n = length3 * (n2 * 0.618 / n_total)
            length1_n = length1 * (n3 / n_total)
            length_total_n = length1_n + length2_n + length3_n

            t_1 = length1_n / length_total_n
            t_2 = (1 - t_1) * 0.382

            kp11 = np.array([0, r_inner + (ra - r_inner) * t_1])
            r_kp11 = kp11[1]

            kp12 = np.array([0, r_inner + (ra - r_inner) * (t_1 + t_2)])
            t14 = 0.382
            kp14 = t14 * kp12 + (1 - t14) * kp6
            kp16 = t14 * kp12 + (1 - t14) * kp8

            # 旋转角
            rot_phi = np.linspace(0, 2 * np.pi, z, endpoint=False)
            # 构造剩余关键点
            kp1 = np.array([r_kp11 * np.cos(angle_kp2), r_kp11 * np.sin(angle_kp2)])
            kp4 = np.array([r_kp11 * np.cos(angle_kp4), r_kp11 * np.sin(angle_kp4)])
            key_points = np.array(
                [kp0, kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12, kp13, kp14, kp15, kp16, kp17])

            # 构造半边数据结构所用分区边
            edge = np.array([[0, 1],
                             [1, 2],
                             [4, 3],
                             [5, 4],
                             [2, 6],
                             [6, 7],
                             [7, 15],
                             [15, 13],
                             [13, 17],
                             [17, 9],
                             [9, 8],
                             [8, 5],
                             [3, 10],
                             [10, 0],
                             [10, 11],
                             [11, 12],
                             [12, 13],
                             [1, 14],
                             [14, 15],
                             [4, 16],
                             [16, 17],
                             [4, 11],
                             [11, 1],
                             [8, 16],
                             [16, 12],
                             [12, 14],
                             [14, 6]])

            # 构建半边数据结构所用边（由点列构成的边）
            na1 = int(na / 2)
            na2 = na - na1
            delta_kp7_15 = np.linspace(angle_kp7, angle_kp15, na1 + 1)
            delta_kp15_13 = np.linspace(angle_kp15, pi / 2, na2 + 1)
            delta_kp13_17 = np.linspace(pi / 2, angle_kp17, na2 + 1)
            delta_kp17_9 = np.linspace(angle_kp17, pi - angle_kp7, na1 + 1)
            delta_kp4_11 = np.linspace(angle_kp4, pi / 2, na2 + 1)
            delta_kp11_1 = np.linspace(pi / 2, angle_kp1, na2 + 1)
            line = [
                np.linspace(key_points[edge[0, 0]], key_points[edge[0, 1]], n3 + 1),
                np.linspace(key_points[edge[1, 0]], key_points[edge[1, 1]], na1 + 1),
                np.linspace(key_points[edge[2, 0]], key_points[edge[2, 1]], n3 + 1),
                np.linspace(key_points[edge[3, 0]], key_points[edge[3, 1]], na1 + 1),
                points[n1 + n2 + 1:n1 + 2 * n2 + 2, :-1],
                points[n1 + 2 * n2 + 1:2 * n1 + 2 * n2 + 2, :-1],
                np.concatenate([ra * np.cos(delta_kp7_15)[:, None], ra * np.sin(delta_kp7_15)[:, None]], axis=1),
                np.concatenate([ra * np.cos(delta_kp15_13)[:, None], ra * np.sin(delta_kp15_13)[:, None]], axis=1),
                np.concatenate([ra * np.cos(delta_kp13_17)[:, None], ra * np.sin(delta_kp13_17)[:, None]], axis=1),
                np.concatenate([ra * np.cos(delta_kp17_9)[:, None], ra * np.sin(delta_kp17_9)[:, None]], axis=1),
                points[n2:n1 + n2 + 1, :-1][::-1],
                points[:n2 + 1, :-1][::-1],
                np.concatenate([r_inner * np.cos(delta_kp4_11)[:, None], r_inner * np.sin(delta_kp4_11)[:, None]],
                               axis=1),
                np.concatenate([r_inner * np.cos(delta_kp11_1)[:, None], r_inner * np.sin(delta_kp11_1)[:, None]],
                               axis=1),
                np.linspace(key_points[edge[14, 0]], key_points[edge[14, 1]], n3 + 1),
                np.linspace(key_points[edge[15, 0]], key_points[edge[15, 1]], n2 + 1),
                np.linspace(key_points[edge[16, 0]], key_points[edge[16, 1]], n1 + 1),
                line_transformer(points[n1 + n2 + 1:n1 + 2 * n2 + 2, :-1], kp1, kp14),
                line_transformer(points[n1 + 2 * n2 + 1:2 * n1 + 2 * n2 + 2, :-1], kp14, kp15),
                line_transformer(points[:n2 + 1, :-1], kp4, kp16),
                line_transformer(points[n2:n1 + n2 + 1, :-1], kp16, kp17),
                np.concatenate([r_kp11 * np.cos(delta_kp4_11)[:, None], r_kp11 * np.sin(delta_kp4_11)[:, None]],
                               axis=1),
                np.concatenate([r_kp11 * np.cos(delta_kp11_1)[:, None], r_kp11 * np.sin(delta_kp11_1)[:, None]],
                               axis=1),
                np.linspace(key_points[edge[23, 0]], key_points[edge[23, 1]], na1 + 1),
                np.linspace(key_points[edge[24, 0]], key_points[edge[24, 1]], na2 + 1),
                np.linspace(key_points[edge[25, 0]], key_points[edge[25, 1]], na2 + 1),
                np.linspace(key_points[edge[26, 0]], key_points[edge[26, 1]], na1 + 1)
            ]

            # 构建子区域半边数据结构
            boundary_edge = np.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 3, 2, 12, 13])
            half_edge = subdomain_divider(line, key_points, edge, boundary_edge)

            # 单齿网格及其节点与单元
            quad_mesh = QuadrangleMesh.sub_domain_mesh_generator(half_edge, key_points, line)
            cell_domain_tag = quad_mesh.celldata['cell_domain_tag']
            tooth_node = quad_mesh.node
            tooth_cell = quad_mesh.cell
            origin_cell = quad_mesh.cell
            cell_cell_num = len(origin_cell)
            cell_tooth_tag = np.zeros(cell_cell_num * z, dtype=np.int_)

            # 旋转构建剩余点与单元，并依次拼接
            single_node_num = len(tooth_node) - (n3 + na1 + 1)
            temp_node = np.concatenate(
                [tooth_node[3:len(key_points)], tooth_node[len(key_points) + (n3 - 1) + (na1 - 1):]], axis=0)
            # 最后一个齿的节点，需要特殊处理
            temp_node_last = np.concatenate(
                [tooth_node[6:len(key_points)], tooth_node[len(key_points) + 2 * (n3 - 1) + 2 * (na1 - 1):]], axis=0)

            # 辅助所用的节点映射，将新节点编号按照初始单元节点排列
            origin_trans_matrix = np.arange(len(tooth_node))
            trans_matrix = np.arange(len(tooth_node))
            # 左侧齿
            # 处理重复顶点
            trans_matrix[0] = trans_matrix[3]
            trans_matrix[1] = trans_matrix[4]
            trans_matrix[2] = trans_matrix[5]
            # 处理重复边上节点
            trans_matrix[len(key_points):len(key_points) + (n3 - 1)] \
                = trans_matrix[len(key_points) + (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + (na1 - 1)][::-1]
            trans_matrix[len(key_points) + (n3 - 1):len(key_points) + (n3 - 1) + (na1 - 1)] \
                = trans_matrix[
                  len(key_points) + 2 * (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + 2 * (na1 - 1)][::-1]
            # 其他节点
            trans_matrix[3:len(key_points)] += single_node_num + (n3 - 1) + (na1 - 1)
            trans_matrix[len(key_points) + (n3 - 1) + (na1 - 1):] += single_node_num
            # 计算新节点与单元
            rot_matrix = np.array([[np.cos(rot_phi[1]), -np.sin(rot_phi[1])], [np.sin(rot_phi[1]), np.cos(rot_phi[1])]])
            new_node = np.einsum('ij,jn->in', rot_matrix, temp_node.T).T
            new_cell = trans_matrix[origin_cell]
            cell_tooth_tag[cell_cell_num:2 * cell_cell_num] = 1
            # 拼接
            tooth_node = np.concatenate([tooth_node, new_node], axis=0)
            tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)
            t_mesh = QuadrangleMesh(tooth_node, tooth_cell)

            # 中间齿
            for i in range(2, z - 1):
                rot_matrix = np.array(
                    [[np.cos(rot_phi[i]), -np.sin(rot_phi[i])], [np.sin(rot_phi[i]), np.cos(rot_phi[i])]])
                new_node = np.einsum('ij,jn->in', rot_matrix, temp_node.T).T
                # 处理重复顶点
                trans_matrix[0] = trans_matrix[3]
                trans_matrix[1] = trans_matrix[4]
                trans_matrix[2] = trans_matrix[5]
                # 处理重复边上节点
                trans_matrix[len(key_points):len(key_points) + (n3 - 1)] \
                    = trans_matrix[len(key_points) + (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + (na1 - 1)][
                      ::-1]
                trans_matrix[len(key_points) + (n3 - 1):len(key_points) + (n3 - 1) + (na1 - 1)] \
                    = trans_matrix[
                      len(key_points) + 2 * (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + 2 * (na1 - 1)][::-1]
                # 其他节点
                trans_matrix[3:len(key_points)] += single_node_num
                trans_matrix[len(key_points) + (n3 - 1) + (na1 - 1):] += single_node_num
                # 新单元映射与拼接
                new_cell = trans_matrix[origin_cell]
                cell_tooth_tag[i * cell_cell_num:(i + 1) * cell_cell_num] = i
                tooth_node = np.concatenate([tooth_node, new_node], axis=0)
                tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)

            # 右侧齿
            rot_matrix = np.array(
                [[np.cos(rot_phi[-1]), -np.sin(rot_phi[-1])], [np.sin(rot_phi[-1]), np.cos(rot_phi[-1])]])
            new_node = np.einsum('ij,jn->in', rot_matrix, temp_node_last.T).T
            # 处理重复顶点
            trans_matrix[0] = trans_matrix[3]
            trans_matrix[1] = trans_matrix[4]
            trans_matrix[2] = trans_matrix[5]
            trans_matrix[3] = origin_trans_matrix[0]
            trans_matrix[4] = origin_trans_matrix[1]
            trans_matrix[5] = origin_trans_matrix[2]
            # 处理重复边上节点
            trans_matrix[len(key_points):len(key_points) + (n3 - 1)] \
                = trans_matrix[len(key_points) + (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + (na1 - 1)][::-1]
            trans_matrix[len(key_points) + (n3 - 1):len(key_points) + (n3 - 1) + (na1 - 1)] \
                = trans_matrix[
                  len(key_points) + 2 * (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + 2 * (na1 - 1)][::-1]
            trans_matrix[len(key_points) + (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + (na1 - 1)] \
                = origin_trans_matrix[len(key_points):len(key_points) + (n3 - 1)][::-1]
            trans_matrix[len(key_points) + 2 * (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + 2 * (na1 - 1)] \
                = origin_trans_matrix[len(key_points) + (n3 - 1):len(key_points) + (n3 - 1) + (na1 - 1)][::-1]
            # 其他节点
            trans_matrix[6:len(key_points)] += single_node_num - 3
            trans_matrix[len(key_points) + 2 * (n3 - 1) + 2 * (na1 - 1):] += single_node_num - (n3 - 1) - (na1 - 1) - 3
            # 新单元映射与拼接
            new_cell = trans_matrix[origin_cell]
            cell_tooth_tag[(z - 1) * cell_cell_num:] = z - 1
            tooth_node = np.concatenate([tooth_node, new_node], axis=0)
            tooth_node = np.dot(tooth_node, np.array([[0, -1], [1, 0]]))
            tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)
            # 最终网格
            t_mesh = QuadrangleMesh(tooth_node, tooth_cell)
            t_mesh.celldata['cell_domain_tag'] = np.tile(cell_domain_tag, z)
            t_mesh.celldata['cell_tooth_tag'] = cell_tooth_tag
        else:
            # 构造关键点
            kp20 = points[n1 + n2 + 1, :2]
            angle_kp20 = np.arctan2(kp20[1], kp20[0])
            delta_angle_kp20 = pi / 2 - angle_kp20
            angle_kp19 = pi / 2 - delta_angle_kp20
            angle_kp22 = pi / 2 + delta_angle_kp20
            kp23 = points[0, :2]
            kp6 = points[n1 + 2 * n2 + 1, :2]
            kp8 = points[n2, :2]
            kp7 = points[-1, :2]
            kp9 = points[n1 + n2, :2]

            angle_kp7 = np.arctan2(kp7[1], kp7[0])
            delta_angle_kp7 = pi / 2 - angle_kp7
            angle_kp15 = pi / 2 - delta_angle_kp7 / 2
            angle_kp17 = pi / 2 + delta_angle_kp7 / 2
            kp15 = np.array([ra * np.cos(angle_kp15), ra * np.sin(angle_kp15)])
            kp17 = np.array([ra * np.cos(angle_kp17), ra * np.sin(angle_kp17)])

            kp18 = np.array([r_inner * np.cos(angle_kp19), r_inner * np.sin(angle_kp19)])
            kp21 = np.array([r_inner * np.cos(angle_kp22), r_inner * np.sin(angle_kp22)])

            kp10 = np.array([0, r_inner])
            kp13 = np.array([0, ra])

            # 单侧弧长与点数，计算中轴上点参数
            distance = np.sqrt(np.sum(np.diff(points[:n1 + n2 + 1, :2], axis=0) ** 2, axis=1))
            length2 = np.sum(distance[:n1])
            length3 = np.sum(distance[n1:n1 + n2])
            length1 = np.sqrt(np.sum((kp23 - kp21) ** 2))

            n_total = n1 * 0.618 + n2 * 0.618 + n3
            length2_n = length2 * (n1 * 0.618 / n_total)
            length3_n = length3 * (n2 * 0.618 / n_total)
            length1_n = length1 * (n3 / n_total)
            length_total_n = length1_n + length2_n + length3_n

            t_1 = length1_n / length_total_n
            t_2 = (1 - t_1) * 0.382

            kp11 = np.array([0, r_inner + (ra - r_inner) * t_1])
            r_kp11 = kp11[1]
            kp19 = np.array([r_kp11 * np.cos(angle_kp19), r_kp11 * np.sin(angle_kp19)])
            kp22 = np.array([r_kp11 * np.cos(angle_kp22), r_kp11 * np.sin(angle_kp22)])

            kp12 = np.array([0, r_inner + (ra - r_inner) * (t_1 + t_2)])
            t14 = 0.382
            kp14 = t14 * kp12 + (1 - t14) * kp6
            kp16 = t14 * kp12 + (1 - t14) * kp8

            # 旋转角
            rot_phi = np.linspace(0, 2 * np.pi, z, endpoint=False)

            # 齿根圆弧上点计算
            rot_kp_1 = np.zeros(2)
            rot_kp_1[0] = np.cos(rot_phi[1]) * kp20[0] - np.sin(rot_phi[1]) * kp20[1]
            rot_kp_1[1] = np.sin(rot_phi[1]) * kp20[0] + np.cos(rot_phi[1]) * kp20[1]
            angle0 = np.arctan2(kp20[1], kp20[0])
            angle1 = np.arctan2(rot_kp_1[1], rot_kp_1[0])
            angle2 = np.arctan2(kp23[1], kp23[0])
            delta_angle = abs(angle1 - angle2)

            # 构造剩余关键点
            kp_0_angle = angle0 - delta_angle / 2
            kp_3_angle = angle2 + delta_angle / 2
            kp0 = np.array([r_inner * np.cos(kp_0_angle), r_inner * np.sin(kp_0_angle)])
            kp1 = np.array([r_kp11 * np.cos(kp_0_angle), r_kp11 * np.sin(kp_0_angle)])
            kp2 = np.array([rf * np.cos(kp_0_angle), rf * np.sin(kp_0_angle)])
            kp3 = np.array([r_inner * np.cos(kp_3_angle), r_inner * np.sin(kp_3_angle)])
            kp4 = np.array([r_kp11 * np.cos(kp_3_angle), r_kp11 * np.sin(kp_3_angle)])
            kp5 = np.array([rf * np.cos(kp_3_angle), rf * np.sin(kp_3_angle)])
            key_points = np.array(
                [kp0, kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12, kp13, kp14, kp15, kp16, kp17, kp18,
                 kp19, kp20, kp21, kp22, kp23])

            # 构造半边数据结构所用分区边
            edge = np.array([[0, 1],
                             [1, 2],
                             [4, 3],
                             [5, 4],
                             [20, 6],
                             [6, 7],
                             [7, 15],
                             [15, 13],
                             [13, 17],
                             [17, 9],
                             [9, 8],
                             [8, 23],
                             [21, 10],
                             [10, 18],
                             [10, 11],
                             [11, 12],
                             [12, 13],
                             [19, 14],
                             [14, 15],
                             [22, 16],
                             [16, 17],
                             [22, 11],
                             [11, 19],
                             [8, 16],
                             [16, 12],
                             [12, 14],
                             [14, 6],
                             [2, 20],
                             [23, 5],
                             [3, 21],
                             [18, 0],
                             [18, 19],
                             [19, 20],
                             [21, 22],
                             [22, 23],
                             [4, 22],
                             [19, 1]])

            # 构建半边数据结构所用边（由点列构成的边）
            na1 = int(na / 2)
            na2 = na - na1

            angle_kp1 = np.arctan2(kp1[1], kp1[0])
            delta_kp7_15 = np.linspace(angle_kp7, angle_kp15, na1 + 1)
            delta_kp15_13 = np.linspace(angle_kp15, pi / 2, na2 + 1)
            delta_kp13_17 = np.linspace(pi / 2, angle_kp17, na2 + 1)
            delta_kp17_9 = np.linspace(angle_kp17, pi - angle_kp7, na1 + 1)
            delta_kp22_11 = np.linspace(angle_kp22, pi / 2, na2 + 1)
            delta_kp11_19 = np.linspace(pi / 2, angle_kp19, na2 + 1)
            delta_kp19_1 = np.linspace(angle_kp19, angle_kp1, nf + 1)
            delta_kp4_22 = np.linspace(pi - angle_kp1, angle_kp22, nf + 1)
            line = [
                np.linspace(key_points[edge[0, 0]], key_points[edge[0, 1]], n3 + 1),
                np.linspace(key_points[edge[1, 0]], key_points[edge[1, 1]], na1 + 1),
                np.linspace(key_points[edge[2, 0]], key_points[edge[2, 1]], n3 + 1),
                np.linspace(key_points[edge[3, 0]], key_points[edge[3, 1]], na1 + 1),
                points[n1 + n2 + 1:n1 + 2 * n2 + 2, :-1],
                points[n1 + 2 * n2 + 1:2 * n1 + 2 * n2 + 2, :-1],
                np.concatenate([ra * np.cos(delta_kp7_15)[:, None], ra * np.sin(delta_kp7_15)[:, None]], axis=1),
                np.concatenate([ra * np.cos(delta_kp15_13)[:, None], ra * np.sin(delta_kp15_13)[:, None]], axis=1),
                np.concatenate([ra * np.cos(delta_kp13_17)[:, None], ra * np.sin(delta_kp13_17)[:, None]], axis=1),
                np.concatenate([ra * np.cos(delta_kp17_9)[:, None], ra * np.sin(delta_kp17_9)[:, None]], axis=1),
                points[n2:n1 + n2 + 1, :-1][::-1],
                points[:n2 + 1, :-1][::-1],
                np.concatenate([r_inner * np.cos(delta_kp22_11)[:, None], r_inner * np.sin(delta_kp22_11)[:, None]],
                               axis=1),
                np.concatenate([r_inner * np.cos(delta_kp11_19)[:, None], r_inner * np.sin(delta_kp11_19)[:, None]],
                               axis=1),
                np.linspace(key_points[edge[14, 0]], key_points[edge[14, 1]], n3 + 1),
                np.linspace(key_points[edge[15, 0]], key_points[edge[15, 1]], n2 + 1),
                np.linspace(key_points[edge[16, 0]], key_points[edge[16, 1]], n1 + 1),
                line_transformer(points[n1 + n2 + 1:n1 + 2 * n2 + 2, :-1], kp19, kp14),
                line_transformer(points[n1 + 2 * n2 + 1:2 * n1 + 2 * n2 + 2, :-1], kp14, kp15),
                line_transformer(points[:n2 + 1, :-1], kp22, kp16),
                line_transformer(points[n2:n1 + n2 + 1, :-1], kp16, kp17),
                np.concatenate([r_kp11 * np.cos(delta_kp22_11)[:, None], r_kp11 * np.sin(delta_kp22_11)[:, None]],
                               axis=1),
                np.concatenate([r_kp11 * np.cos(delta_kp11_19)[:, None], r_kp11 * np.sin(delta_kp11_19)[:, None]],
                               axis=1),
                np.linspace(key_points[edge[23, 0]], key_points[edge[23, 1]], na1 + 1),
                np.linspace(key_points[edge[24, 0]], key_points[edge[24, 1]], na2 + 1),
                np.linspace(key_points[edge[25, 0]], key_points[edge[25, 1]], na2 + 1),
                np.linspace(key_points[edge[26, 0]], key_points[edge[26, 1]], na1 + 1),
                np.concatenate([rf * np.cos(delta_kp19_1)[:, None], rf * np.sin(delta_kp19_1)[:, None]], axis=1)[::-1],
                np.concatenate([rf * np.cos(delta_kp4_22)[:, None], rf * np.sin(delta_kp4_22)[:, None]], axis=1)[::-1],
                np.concatenate([r_inner * np.cos(delta_kp4_22)[:, None], r_inner * np.sin(delta_kp4_22)[:, None]],
                               axis=1),
                np.concatenate([r_inner * np.cos(delta_kp19_1)[:, None], r_inner * np.sin(delta_kp19_1)[:, None]],
                               axis=1),
                np.linspace(key_points[edge[31, 0]], key_points[edge[31, 1]], n3 + 1),
                np.linspace(key_points[edge[32, 0]], key_points[edge[32, 1]], na1 + 1),
                np.linspace(key_points[edge[33, 0]], key_points[edge[33, 1]], n3 + 1),
                np.linspace(key_points[edge[34, 0]], key_points[edge[34, 1]], na1 + 1),
                np.concatenate([r_kp11 * np.cos(delta_kp4_22)[:, None], r_kp11 * np.sin(delta_kp4_22)[:, None]],
                               axis=1),
                np.concatenate([r_kp11 * np.cos(delta_kp19_1)[:, None], r_kp11 * np.sin(delta_kp19_1)[:, None]],
                               axis=1),
            ]

            # 构建子区域半边数据结构
            boundary_edge = np.array([0, 1, 27, 4, 5, 6, 7, 8, 9, 10, 11, 28, 3, 2, 29, 12, 13, 30])
            half_edge = subdomain_divider(line, key_points, edge, boundary_edge)

            # 单齿网格及其节点与单元
            quad_mesh = QuadrangleMesh.sub_domain_mesh_generator(half_edge, key_points, line)
            cell_domain_tag = quad_mesh.celldata['cell_domain_tag']
            tooth_node = quad_mesh.node
            tooth_cell = quad_mesh.cell
            origin_cell = quad_mesh.cell
            cell_cell_num = len(origin_cell)
            cell_tooth_tag = np.zeros(cell_cell_num*z, dtype=np.int_)
            # 旋转构建剩余点与单元，并依次拼接
            single_node_num = len(tooth_node) - (n3 + na1 + 1)
            temp_node = np.concatenate(
                [tooth_node[3:len(key_points)], tooth_node[len(key_points) + (n3 - 1) + (na1 - 1):]], axis=0)
            # 最后一个齿的节点，需要特殊处理
            temp_node_last = np.concatenate(
                [tooth_node[6:len(key_points)], tooth_node[len(key_points) + 2 * (n3 - 1) + 2 * (na1 - 1):]], axis=0)
            # 辅助所用的节点映射，将新节点编号按照初始单元节点排列
            origin_trans_matrix = np.arange(len(tooth_node))
            trans_matrix = np.arange(len(tooth_node))
            # 左侧齿
            # 处理重复顶点
            trans_matrix[0] = trans_matrix[3]
            trans_matrix[1] = trans_matrix[4]
            trans_matrix[2] = trans_matrix[5]
            # 处理重复边上节点
            trans_matrix[len(key_points):len(key_points) + (n3 - 1)] \
                = trans_matrix[len(key_points) + (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + (na1 - 1)][::-1]
            trans_matrix[len(key_points) + (n3 - 1):len(key_points) + (n3 - 1) + (na1 - 1)] \
                = trans_matrix[
                  len(key_points) + 2 * (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + 2 * (na1 - 1)][::-1]
            # 其他节点
            trans_matrix[3:len(key_points)] += single_node_num + (n3 - 1) + (na1 - 1)
            trans_matrix[len(key_points) + (n3 - 1) + (na1 - 1):] += single_node_num
            # 计算新节点与单元
            rot_matrix = np.array([[np.cos(rot_phi[1]), -np.sin(rot_phi[1])], [np.sin(rot_phi[1]), np.cos(rot_phi[1])]])
            new_node = np.einsum('ij,jn->in', rot_matrix, temp_node.T).T
            new_cell = trans_matrix[origin_cell]
            cell_tooth_tag[cell_cell_num:2*cell_cell_num] = 1
            # 拼接
            tooth_node = np.concatenate([tooth_node, new_node], axis=0)
            tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)

            # 中间齿
            for i in range(2, z - 1):
                rot_matrix = np.array(
                    [[np.cos(rot_phi[i]), -np.sin(rot_phi[i])], [np.sin(rot_phi[i]), np.cos(rot_phi[i])]])
                new_node = np.einsum('ij,jn->in', rot_matrix, temp_node.T).T
                # 处理重复顶点
                trans_matrix[0] = trans_matrix[3]
                trans_matrix[1] = trans_matrix[4]
                trans_matrix[2] = trans_matrix[5]
                # 处理重复边上节点
                trans_matrix[len(key_points):len(key_points) + (n3 - 1)] \
                    = trans_matrix[len(key_points) + (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + (na1 - 1)][
                      ::-1]
                trans_matrix[len(key_points) + (n3 - 1):len(key_points) + (n3 - 1) + (na1 - 1)] \
                    = trans_matrix[
                      len(key_points) + 2 * (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + 2 * (na1 - 1)][::-1]
                # 其他节点
                trans_matrix[3:len(key_points)] += single_node_num
                trans_matrix[len(key_points) + (n3 - 1) + (na1 - 1):] += single_node_num
                # 新单元映射与拼接
                new_cell = trans_matrix[origin_cell]
                cell_tooth_tag[i*cell_cell_num:(i+1)*cell_cell_num] = i
                tooth_node = np.concatenate([tooth_node, new_node], axis=0)
                tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)

            # 右侧齿
            rot_matrix = np.array(
                [[np.cos(rot_phi[-1]), -np.sin(rot_phi[-1])], [np.sin(rot_phi[-1]), np.cos(rot_phi[-1])]])
            new_node = np.einsum('ij,jn->in', rot_matrix, temp_node_last.T).T
            # 处理重复顶点
            trans_matrix[0] = trans_matrix[3]
            trans_matrix[1] = trans_matrix[4]
            trans_matrix[2] = trans_matrix[5]
            trans_matrix[3] = origin_trans_matrix[0]
            trans_matrix[4] = origin_trans_matrix[1]
            trans_matrix[5] = origin_trans_matrix[2]
            # 处理重复边上节点
            trans_matrix[len(key_points):len(key_points) + (n3 - 1)] \
                = trans_matrix[len(key_points) + (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + (na1 - 1)][::-1]
            trans_matrix[len(key_points) + (n3 - 1):len(key_points) + (n3 - 1) + (na1 - 1)] \
                = trans_matrix[
                  len(key_points) + 2 * (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + 2 * (na1 - 1)][::-1]
            trans_matrix[len(key_points) + (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + (na1 - 1)] \
                = origin_trans_matrix[len(key_points):len(key_points) + (n3 - 1)][::-1]
            trans_matrix[len(key_points) + 2 * (n3 - 1) + (na1 - 1):len(key_points) + 2 * (n3 - 1) + 2 * (na1 - 1)] \
                = origin_trans_matrix[len(key_points) + (n3 - 1):len(key_points) + (n3 - 1) + (na1 - 1)][::-1]
            # 其他节点
            trans_matrix[6:len(key_points)] += single_node_num - 3
            trans_matrix[len(key_points) + 2 * (n3 - 1) + 2 * (na1 - 1):] += single_node_num - (n3 - 1) - (na1 - 1) - 3
            # 新单元映射与拼接
            new_cell = trans_matrix[origin_cell]
            cell_tooth_tag[(z-1) * cell_cell_num:] = z-1
            tooth_node = np.concatenate([tooth_node, new_node], axis=0)
            tooth_node = np.dot(tooth_node, np.array([[0, -1], [1, 0]]))
            tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)
            # 最终网格
            t_mesh = QuadrangleMesh(tooth_node, tooth_cell)
            t_mesh.celldata['cell_domain_tag'] = np.tile(cell_domain_tag, z)
            t_mesh.celldata['cell_tooth_tag'] = cell_tooth_tag

        self.mesh = t_mesh
        return t_mesh


class InternalGear(Gear):
    def __init__(self, m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, nw, outer_diam, z_cutter,
                 xn_cutter, tooth_width, material=None, rotation_direction=1, center=(0,0,0), name=None):
        """

        @param m_n: 法向模数
        @param z: 齿数
        @param alpha_n: 法向压力角
        @param beta: 螺旋角
        @param x_n: 法向变位系数
        @param hac: 齿顶高系数
        @param cc: 顶隙系数
        @param rcc: 刀尖圆弧半径系数
        @param jn: 法向侧隙
        @param n1: 渐开线分段数
        @param n2: 过渡曲线分段数
        @param n3: 齿轮内部分段数
        @param na: 齿顶分段数
        @param nf: 齿根圆部分分段数（一侧，非最大圆角时）
        @param nw: 沿齿宽方向分段数
        @param outer_diam: 轮缘外径
        @param z_cutter: 刀具齿数
        @param xn_cutter: 刀具变位系数
        @param tooth_width: 齿宽
        @param material: 齿轮材料
        @param rotation_direction: 旋转方向，1 为右旋齿轮，-1 为左旋齿轮，默认为右旋
        @param center: 齿轮中心坐标，默认为原点
        @param name: 齿轮名称
        """
        super().__init__(m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, nw, tooth_width, material, rotation_direction, center, name)
        self.outer_diam = outer_diam
        self.z_cutter = z_cutter
        self.xn_cutter = xn_cutter
        # 齿顶圆直径与半径
        ha = self.m_n * (self.hac - self.x_n)
        self.d_a = self.d - 2 * ha
        self.r_a = self.d_a / 2
        # 齿根圆直径与半径
        hf = self.m_n * (self.hac + self.cc + self.x_n)
        self.d_f = self.d + 2 * hf
        self.r_f = self.d_f / 2
        # 刀具分度圆直径与半径
        self.d_cutter = self.m_t * self.z_cutter
        self.r_cutter = self.d_cutter / 2
        # 刀具基圆直径与半径
        self.db_cutter = self.d_cutter * cos(self.alpha_t)
        self.rb_cutter = self.db_cutter / 2
        # 刀具齿顶圆直径与半径
        ha_cutter = self.m_n * (self.hac + self.cc + self.xn_cutter)
        self.da_cutter = self.d_cutter + 2 * ha_cutter
        self.ra_cutter = self.da_cutter / 2
        # 刀具齿根圆直径与半径
        hf_cutter = self.m_n * (self.hac - self.xn_cutter)
        self.df_cutter = self.d_cutter - 2 * hf_cutter
        self.rf_cutter = self.df_cutter / 2
        # 刀具齿槽半角
        eta = (pi - 4 * self.xn_cutter * tan(self.alpha_n)) / (2 * self.z_cutter)
        self.etab = pi / self.z_cutter - (eta - (tan(self.alpha_t) - self.alpha_t))
        # 刀尖圆弧半径
        self.rc = self.m_n * self.rcc
        # 相关参数计算
        etab = self.etab
        func = lambda t: [
            self.rb_cutter * cos(etab) * (sin(t[0]) - t[0] * cos(t[0])) - self.rb_cutter * sin(etab) * (
                    cos(t[0]) + t[0] * sin(t[0])) - t[
                2] * cos(t[1]),
            self.rb_cutter * cos(etab) * (cos(t[0]) + t[0] * sin(t[0])) + self.rb_cutter * sin(etab) * (
                    sin(t[0]) - t[0] * cos(t[0])) - (
                    t[2] * sin(t[1]) + self.ra_cutter - t[2]),
            (self.rb_cutter * t[0] * sin(etab) * sin(t[0]) + self.rb_cutter * t[0] * cos(etab) * cos(t[0])) / (
                    self.rb_cutter * t[0] * cos(etab) * sin(t[0]) - self.rb_cutter * t[0] * cos(t[0]) * sin(
                etab)) + cos(t[1]) / sin(t[1])
        ]
        self.t = fsolve(func, [1, 0.75 * pi, 0.25 * self.m_n])

    @classmethod
    def ainv(cls, x):
        temp = 0
        alpha = arctan(x)
        while np.abs(alpha - temp) > 1e-25:
            temp = alpha
            alpha = np.arctan(x + temp)
        return alpha

    def get_involute_points(self, t):
        alphan = self.alpha_n
        alphat = self.alpha_t
        rb = self.r_b
        xn = self.x_n
        z = self.z

        eta = (pi - 4 * xn * np.tan(alphan)) / (2 * z)
        etab = pi / z - (eta - (tan(alphat) - alphat))
        xt = (rb * cos(etab) * (sin(t) - t * cos(t)) - rb * sin(etab) * (cos(t) + t * sin(t))).reshape((-1, 1))
        yt = (rb * cos(etab) * (cos(t) + t * sin(t)) + rb * sin(etab) * (sin(t) - t * cos(t))).reshape((-1, 1))

        points = np.concatenate([xt, yt], axis=1)
        return points

    def get_tip_intersection_points(self, t):

        points = self.get_involute_points(t)
        return np.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2)

    def get_transition_points(self, E, x0, y0, rc, ratio, t):

        def calc_phi(t_values):
            def f(phi, tt):
                cos_phi = cos(phi * (ratio - 1))
                sin_phi = sin(phi * (ratio - 1))
                term1 = -(rc * cos_phi * cos(tt) + rc * sin_phi * sin(tt)) * \
                        (E * cos(phi) + sin_phi * (x0 + rc * cos(tt)) * (ratio - 1) - \
                         cos_phi * (y0 + rc * sin(tt)) * (ratio - 1))
                term2 = -(rc * cos_phi * sin(tt) - rc * sin_phi * cos(tt)) * \
                        (E * sin(phi) + cos_phi * (x0 + rc * cos(tt)) * (ratio - 1) + \
                         sin_phi * (y0 + rc * sin(tt)) * (ratio - 1))
                return term1 + term2

            phi_values = []
            for tt in t_values:
                phi = fsolve(f, 0, args=(tt))[0]  # 对每个 tt 调用 fsolve
                phi_values.append(phi)

            return np.array(phi_values)

        phi = calc_phi(t)
        xt = (sin(phi * (ratio - 1)) * (y0 + rc * sin(t)) - E * sin(phi) + cos(phi * (ratio - 1)) * (
                x0 + rc * cos(t))).reshape((-1, 1))
        yt = (E * cos(phi) - sin(phi * (ratio - 1)) * (x0 + rc * cos(t)) + cos(phi * (ratio - 1)) * (
                y0 + rc * sin(t))).reshape((-1, 1))

        points = np.concatenate([xt, yt], axis=1)
        return points

    def get_transition_intersection_points(self, E, x0, y0, rc, ratio, t):
        points = self.get_transition_points(E, x0, y0, rc, ratio, t)
        return np.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2)

    def get_profile_points(self):
        ra_cutter = self.ra_cutter
        rb_cutter = self.rb_cutter
        t = self.t
        rc = self.rc
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3
        na = self.na
        etab = self.etab

        ratio = self.z / self.z_cutter
        alphawt = InternalGear.ainv(
            2 * (self.x_n - self.xn_cutter) * tan(self.alpha_n) / (self.z - self.z_cutter) + (
                    tan(self.alpha_t) - self.alpha_t))
        E = 0.5 * (self.d - self.d_cutter) + self.m_t * (
                0.5 * (self.z - self.z_cutter) * (cos(self.alpha_t) / cos(alphawt) - 1))
        df11 = 2 * (E + ra_cutter)

        if rc >= t[2]:
            points = np.zeros((2 * (n1 + n2) + 1, 3))
            x0 = 0
            y0 = ra_cutter - rc
            t2 = t[1]
            jb = (t[1] - pi / 2) / n2
            tt = np.linspace(t[1], pi / 2, n2, endpoint=False)
            points[n1:n1 + n2, 0:2] = self.get_transition_points(E, x0, y0, rc, ratio, tt)

            # 中点
            points[n1 + n2, 0] = 0
            points[n1 + n2, 1] = 0.5 * df11

            func2 = lambda t: self.get_tip_intersection_points(t) - sqrt(
                points[n1, 0] ** 2 + points[n1, 1] ** 2)
            t2 = fsolve(func2, 1)[0]
            func1 = lambda t: self.get_tip_intersection_points(t) - 0.5 * self.d_a
            t1 = fsolve(func1, 1)[0]
            tt = np.linspace(t1, t2, n1, endpoint=False)
            points[:n1, 0:2] = self.get_involute_points(tt)

            # 对称构造另一侧点
            points[n1 + n2 + 1:, 0] = -points[0:n1 + n2, 0][::-1]
            points[n1 + n2 + 1:, 1] = points[0:n1 + n2, 1][::-1]
        else:
            nf = self.nf
            points = np.zeros((2 * (n1 + n2 + nf) + 1, 3))
            func = lambda t: [
                rb_cutter * cos(etab) * (sin(t[0]) - t[0] * cos(t[0])) - rb_cutter * sin(etab) * (
                        cos(t[0]) + t[0] * sin(t[0])) - (
                        rc * cos(t[1]) + t[2]),
                rb_cutter * cos(etab) * (cos(t[0]) + t[0] * sin(t[0])) + rb_cutter * sin(etab) * (
                        sin(t[0]) - t[0] * cos(t[0])) - (
                        rc * sin(t[1]) + t[3]),
                (rb_cutter * t[0] * sin(etab) * sin(t[0]) + rb_cutter * t[0] * cos(etab) * cos(t[0])) / (
                        rb_cutter * t[0] * cos(etab) * sin(t[0]) - rb_cutter * t[0] * cos(t[0]) * sin(etab)) + cos(
                    t[1]) / sin(t[1]),
                t[2] ** 2 + t[3] ** 2 - (ra_cutter - rc) ** 2
            ]

            t = fsolve(func, [1, 0.75 * pi, 0, ra_cutter])
            self.t = t
            x0 = t[2]
            y0 = t[3]
            func4 = lambda t: self.get_transition_intersection_points(E, x0, y0, rc, ratio, t) - self.r_f

            t3 = pi / 2 - arctan(x0 / y0)
            # t3 = fsolve(func4, pi / 2 - arctan(x0 / y0))[0]
            t4 = t[1]

            tt = np.linspace(t4, t3, n2, endpoint=False)
            points[n1 + 1:n1 + n2 + 1, 0:2] = self.get_transition_points(E, x0, y0, rc, ratio, tt)

            func3 = lambda t: self.get_transition_intersection_points(E, 0, 0, ra_cutter, ratio, t) - self.r_f
            t5 = pi / 2 - arctan(x0 / y0)
            # t5 = fsolve(func3, pi / 2 - arctan(x0 / y0))[0]
            t6 = pi / 2
            tt = np.linspace(t5, t6, nf - 1, endpoint=False)
            points[n1 + n2 + 1:n1 + n2 + nf, 0:2] = self.get_transition_points(E, 0, 0, ra_cutter, ratio, tt)

            points[n1 + n2 + nf, 0] = 0
            points[n1 + n2 + nf, 1] = 0.5 * df11

            func2 = lambda t: self.get_tip_intersection_points(t) - sqrt(
                points[n1 + 1, 0] ** 2 + points[n1 + 1, 1] ** 2)
            t2 = fsolve(func2, 1)[0]

            func1 = lambda t: self.get_tip_intersection_points(t) - 0.5 * self.d_a
            t1 = fsolve(func1, 1)[0]

            tt = np.linspace(t1, t2, n1, endpoint=False)
            points[0:n1, 0:2] = self.get_involute_points(tt)

            points[n1, :] = (points[n1 - 1, :] + points[n1 + 1, :]) / 2

            points[n1 + n2 + nf + 1:, 0] = -points[0:n1 + n2 + nf, 0][::-1]
            points[n1 + n2 + nf + 1:, 1] = points[0:n1 + n2 + nf, 1][::-1]

        return points

    def generate_mesh(self):
        rc = self.rc
        t = self.t
        z = self.z
        ra = self.r_a
        rf = self.r_f
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3
        na = self.na
        nf = self.nf
        outer_diam = self.outer_diam

        if rc >= t[2]:
            points = self.get_profile_points()
            # 构建关键点
            angle_kp0 = pi / 2 - 2 * pi / z / 2
            angle_kp4 = pi / 2 + 2 * pi / z / 2
            t1 = (self.r - ra) / (self.r_f - ra)
            t2 = (self.r_f - ra) / (outer_diam / 2 - ra) * 1.236

            kp0 = ra * np.array([cos(angle_kp0), sin(angle_kp0)]).reshape(1, -1)
            kp3 = outer_diam / 2 * np.array([cos(angle_kp0), sin(angle_kp0)]).reshape(1, -1)
            kp1 = (1 - t1) * kp0 + t1 * kp3
            kp2 = (1 - t2) * kp0 + t2 * kp3

            kp4 = np.zeros_like(kp0)
            kp4[:, 0] = -kp0[:, 0]
            kp4[:, 1] = kp0[:, 1]
            kp5 = np.zeros_like(kp0)
            kp5[:, 0] = -kp1[:, 0]
            kp5[:, 1] = kp1[:, 1]
            kp6 = np.zeros_like(kp0)
            kp6[:, 0] = -kp2[:, 0]
            kp6[:, 1] = kp2[:, 1]
            kp7 = np.zeros_like(kp0)
            kp7[:, 0] = -kp3[:, 0]
            kp7[:, 1] = kp3[:, 1]

            kp10 = points[0:1, 0:2]
            kp11 = points[n1:n1 + 1, 0:2]
            kp14 = points[n1 + n2:n1 + n2 + 1, 0:2]
            kp8 = points[-1:, 0:2]
            kp9 = points[n1 + 2 * n2:n1 + 2 * n2 + 1, 0:2]

            angle_kp8 = np.arctan2(kp8[:, 1], kp8[:, 0])
            angle_kp10 = np.arctan2(kp10[:, 1], kp10[:, 0])
            angle_kp12 = (angle_kp0 + angle_kp8) / 2
            angle_kp13 = pi - angle_kp12

            kp13 = ra * np.array([cos(angle_kp13), sin(angle_kp13)]).reshape(1, -1)
            kp12 = ra * np.array([cos(angle_kp12), sin(angle_kp12)]).reshape(1, -1)

            kp15 = np.array([0, outer_diam / 2]).reshape(1, -1)
            r_kp2 = np.sqrt(np.sum(kp2 ** 2))
            t_kp16 = 0.382
            kp16 = (1 - t_kp16) * kp11 + t_kp16 * kp5
            kp18 = (1 - t_kp16) * kp9 + t_kp16 * kp1
            kp17 = np.array([0, r_kp2]).reshape(1, -1)

            key_points = np.concatenate(
                [kp0, kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12, kp13, kp14, kp15, kp16, kp17, kp18], axis=0)

            # 构造 edge 与 line
            edge = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [5, 4],
                             [6, 5],
                             [7, 6],
                             [3, 15],
                             [15, 7],
                             [4, 13],
                             [13, 10],
                             [10, 11],
                             [11, 14],
                             [14, 9],
                             [9, 8],
                             [8, 12],
                             [12, 0],
                             [12, 18],
                             [18, 17],
                             [13, 16],
                             [16, 17],
                             [14, 17],
                             [17, 15],
                             [5, 16],
                             [16, 11],
                             [9, 18],
                             [18, 1],
                             [6, 17],
                             [17, 2]
                             ])

            na1 = int(na / 2)
            na2 = na - na1
            delta_kp12_0 = np.linspace(angle_kp12, angle_kp0, na2 + 1)
            delta_kp8_12 = np.linspace(angle_kp8, angle_kp12, na1 + 1)
            delta_kp13_10 = np.linspace(angle_kp13, angle_kp10, na1 + 1)
            delta_kp4_13 = np.linspace(angle_kp4, angle_kp13, na2 + 1)
            delta_kp17_2 = np.linspace(pi/2, angle_kp0, na2 + 1).reshape(-1, 1)
            delta_kp6_17 = np.linspace(angle_kp4, pi/2, na2 + 1).reshape(-1, 1)
            delta_kp3_15 = delta_kp17_2[::-1]
            delta_kp15_7 = delta_kp6_17[::-1]
            line = [
                np.linspace(key_points[edge[0, 0]], key_points[edge[0, 1]], n1 + 1),
                np.linspace(key_points[edge[1, 0]], key_points[edge[1, 1]], n2 + 1),
                np.linspace(key_points[edge[2, 0]], key_points[edge[2, 1]], n3 + 1),
                np.linspace(key_points[edge[3, 0]], key_points[edge[3, 1]], n1 + 1),
                np.linspace(key_points[edge[4, 0]], key_points[edge[4, 1]], n2 + 1),
                np.linspace(key_points[edge[5, 0]], key_points[edge[5, 1]], n3 + 1),
                outer_diam / 2 * np.concatenate([cos(delta_kp3_15), sin(delta_kp3_15)], axis=1),
                outer_diam / 2 * np.concatenate([cos(delta_kp15_7), sin(delta_kp15_7)], axis=1),
                ra * np.concatenate([cos(delta_kp4_13), sin(delta_kp4_13)], axis=1),
                ra * np.concatenate([cos(delta_kp13_10), sin(delta_kp13_10)], axis=1),
                points[0:n1 + 1, :-1],
                points[n1:n1 + n2 + 1, :-1],
                points[n1 + n2:n1 + 2 * n2 + 1, :-1],
                points[n1 + 2 * n2:, :-1],
                ra * np.concatenate([cos(delta_kp8_12), sin(delta_kp8_12)], axis=1),
                ra * np.concatenate([cos(delta_kp12_0), sin(delta_kp12_0)], axis=1),
                line_transformer(points[n1 + 2 * n2:, :-1][::-1], kp12, kp18),
                line_transformer(points[n1 + n2:n1 + 2 * n2 + 1, :-1][::-1], kp18, kp17),
                line_transformer(points[0:n1 + 1, :-1], kp13, kp16),
                line_transformer(points[n1:n1 + n2 + 1, :-1], kp16, kp17),
                np.linspace(key_points[edge[20, 0]], key_points[edge[20, 1]], na1 + 1),
                np.linspace(key_points[edge[21, 0]], key_points[edge[21, 1]], n3 + 1),
                np.linspace(key_points[edge[22, 0]], key_points[edge[22, 1]], na2 + 1),
                np.linspace(key_points[edge[23, 0]], key_points[edge[23, 1]], na1 + 1),
                np.linspace(key_points[edge[24, 0]], key_points[edge[24, 1]], na1 + 1),
                np.linspace(key_points[edge[25, 0]], key_points[edge[25, 1]], na2 + 1),
                r_kp2 * np.concatenate([cos(delta_kp6_17), sin(delta_kp6_17)], axis=1),
                r_kp2 * np.concatenate([cos(delta_kp17_2), sin(delta_kp17_2)], axis=1)
            ]

            boundary_edge = np.array([0, 1, 2, 6, 7, 5, 4, 3, 8, 9, 10, 11, 12, 13, 14, 15])
            half_edge = subdomain_divider(line, key_points, edge, boundary_edge)

            quad_mesh = QuadrangleMesh.sub_domain_mesh_generator(half_edge, key_points, line)
            cell_domain_tag = quad_mesh.celldata['cell_domain_tag']
            tooth_node = quad_mesh.node
            tooth_cell = quad_mesh.cell
            origin_cell = quad_mesh.cell
            cell_cell_num = len(origin_cell)
            cell_tooth_tag = np.zeros(cell_cell_num * z, dtype=np.int_)

            # 旋转角
            rot_phi = np.linspace(0, 2 * np.pi, z, endpoint=False)

            # 生成完整内齿
            single_node_num = len(tooth_node) - (n1 + n2 + n3 + 1)
            single_cell_num = len(tooth_cell)
            temp_node = np.concatenate(
                [tooth_node[4:len(key_points)], tooth_node[len(key_points) + (n1 + n2 + n3 - 3):]], axis=0)
            temp_node_last = np.concatenate(
                [tooth_node[8:len(key_points)], tooth_node[len(key_points) + 2 * (n1 + n2 + n3 - 3):]], axis=0)
            origin_trans_matrix = np.arange(len(tooth_node))
            trans_matrix = np.arange(len(tooth_node))
            # 左侧齿
            # 处理重复顶点
            trans_matrix[0] = trans_matrix[4]
            trans_matrix[1] = trans_matrix[5]
            trans_matrix[2] = trans_matrix[6]
            trans_matrix[3] = trans_matrix[7]
            # 处理重复边上节点
            trans_matrix[len(key_points):len(key_points) + (n1 - 1)] \
                = trans_matrix[
                  len(key_points) + (n1 + n2 + n3 - 3):len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1)][::-1]
            trans_matrix[len(key_points) + (n1 - 1):len(key_points) + (n1 + n2 - 2)] \
                = trans_matrix[
                  len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1):len(key_points) + 2 * (n1 + n2 - 2) + (n3 - 1)][
                  ::-1]
            trans_matrix[len(key_points) + (n1 + n2 - 2):len(key_points) + (n1 + n2 + n3 - 3)] \
                = trans_matrix[len(key_points) + 2 * (n1 + n2 - 2) + (n3 - 1):len(key_points) + 2 * (n1 + n2 + n3 - 3)][
                  ::-1]

            # 其他节点
            trans_matrix[4:len(key_points)] += single_node_num + (n1 + n2 + n3 - 3)
            trans_matrix[len(key_points) + (n1 + n2 + n3 - 3):] += single_node_num

            rot_matrix = np.array([[np.cos(rot_phi[1]), -np.sin(rot_phi[1])], [np.sin(rot_phi[1]), np.cos(rot_phi[1])]])
            new_node = np.einsum('ij,jn->in', rot_matrix, temp_node.T).T
            new_cell = trans_matrix[origin_cell]
            cell_tooth_tag[cell_cell_num:2 * cell_cell_num] = 1

            tooth_node = np.concatenate([tooth_node, new_node], axis=0)
            tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)

            for i in range(2, z - 1):
                rot_matrix = np.array(
                    [[np.cos(rot_phi[i]), -np.sin(rot_phi[i])], [np.sin(rot_phi[i]), np.cos(rot_phi[i])]])
                new_node = np.einsum('ij,jn->in', rot_matrix, temp_node.T).T

                # 处理重复顶点
                trans_matrix[0] = trans_matrix[4]
                trans_matrix[1] = trans_matrix[5]
                trans_matrix[2] = trans_matrix[6]
                trans_matrix[3] = trans_matrix[7]
                # 处理重复边上节点
                trans_matrix[len(key_points):len(key_points) + (n1 - 1)] \
                    = trans_matrix[
                      len(key_points) + (n1 + n2 + n3 - 3):len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1)][::-1]
                trans_matrix[len(key_points) + (n1 - 1):len(key_points) + (n1 + n2 - 2)] \
                    = trans_matrix[
                      len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1):len(key_points) + 2 * (n1 + n2 - 2) + (
                              n3 - 1)][::-1]
                trans_matrix[len(key_points) + (n1 + n2 - 2):len(key_points) + (n1 + n2 + n3 - 3)] \
                    = trans_matrix[
                      len(key_points) + 2 * (n1 + n2 - 2) + (n3 - 1):len(key_points) + 2 * (n1 + n2 + n3 - 3)][::-1]
                # 其他节点
                trans_matrix[4:len(key_points)] += single_node_num
                trans_matrix[len(key_points) + (n1 + n2 + n3 - 3):] += single_node_num

                new_cell = trans_matrix[origin_cell]
                cell_tooth_tag[i * cell_cell_num:(i + 1) * cell_cell_num] = i
                tooth_node = np.concatenate([tooth_node, new_node], axis=0)
                tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)

            # 右侧齿
            rot_matrix = np.array(
                [[np.cos(rot_phi[-1]), -np.sin(rot_phi[-1])], [np.sin(rot_phi[-1]), np.cos(rot_phi[-1])]])
            new_node = np.einsum('ij,jn->in', rot_matrix, temp_node_last.T).T
            # 处理重复顶点
            trans_matrix[0] = trans_matrix[4]
            trans_matrix[1] = trans_matrix[5]
            trans_matrix[2] = trans_matrix[6]
            trans_matrix[3] = trans_matrix[7]
            trans_matrix[4] = origin_trans_matrix[0]
            trans_matrix[5] = origin_trans_matrix[1]
            trans_matrix[6] = origin_trans_matrix[2]
            trans_matrix[7] = origin_trans_matrix[3]
            # 处理重复边上节点
            trans_matrix[len(key_points):len(key_points) + (n1 - 1)] \
                = trans_matrix[
                  len(key_points) + (n1 + n2 + n3 - 3):len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1)][::-1]
            trans_matrix[len(key_points) + (n1 - 1):len(key_points) + (n1 + n2 - 2)] \
                = trans_matrix[
                  len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1):len(key_points) + 2 * (n1 + n2 - 2) + (n3 - 1)][
                  ::-1]
            trans_matrix[len(key_points) + (n1 + n2 - 2):len(key_points) + (n1 + n2 + n3 - 3)] \
                = trans_matrix[len(key_points) + 2 * (n1 + n2 - 2) + (n3 - 1):len(key_points) + 2 * (n1 + n2 + n3 - 3)][
                  ::-1]
            trans_matrix[len(key_points) + (n1 + n2 + n3 - 3):len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (
                    n3 - 1)] = origin_trans_matrix[len(key_points):len(key_points) + (n1 - 1)][::-1]
            trans_matrix[len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1):len(key_points) + 2 * (n1 + n2 - 2) + (
                    n3 - 1)] = origin_trans_matrix[len(key_points) + (n1 - 1):len(key_points) + (n1 + n2 - 2)][::-1]
            trans_matrix[len(key_points) + 2 * (n1 + n2 - 2) + (n3 - 1):len(key_points) + 2 * (
                    n1 + n2 + n3 - 3)] = origin_trans_matrix[
                                         len(key_points) + (n1 + n2 - 2):len(key_points) + (n1 + n2 + n3 - 3)][::-1]
            # 其他节点
            trans_matrix[8:len(key_points)] += single_node_num - 4
            trans_matrix[len(key_points) + 2 * (n1 + n2 + n3 - 3):] += single_node_num - (n1 + n2 + n3 + 1)

            new_cell = trans_matrix[origin_cell]
            cell_tooth_tag[(z - 1) * cell_cell_num:] = z - 1
            tooth_node = np.concatenate([tooth_node, new_node], axis=0)
            tooth_node = np.dot(tooth_node, np.array([[0, -1], [1, 0]]))
            tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)

            t_mesh = QuadrangleMesh(tooth_node, tooth_cell)
            t_mesh.celldata['cell_domain_tag'] = np.tile(cell_domain_tag, z)
            t_mesh.celldata['cell_tooth_tag'] = cell_tooth_tag
        else:
            points = self.get_profile_points()
            # 构建关键点
            angle_kp0 = pi / 2 - 2 * pi / z / 2
            angle_kp4 = pi / 2 + 2 * pi / z / 2
            t1 = (self.r - ra) / (self.r_f - ra)
            t2 = (self.r_f - ra) / (outer_diam / 2 - ra) * 1.236

            kp0 = ra * np.array([cos(angle_kp0), sin(angle_kp0)]).reshape(1, -1)
            kp3 = outer_diam / 2 * np.array([cos(angle_kp0), sin(angle_kp0)]).reshape(1, -1)
            kp1 = (1 - t1) * kp0 + t1 * kp3
            kp2 = (1 - t2) * kp0 + t2 * kp3

            kp4 = np.zeros_like(kp0)
            kp4[:, 0] = -kp0[:, 0]
            kp4[:, 1] = kp0[:, 1]
            kp5 = np.zeros_like(kp0)
            kp5[:, 0] = -kp1[:, 0]
            kp5[:, 1] = kp1[:, 1]
            kp6 = np.zeros_like(kp0)
            kp6[:, 0] = -kp2[:, 0]
            kp6[:, 1] = kp2[:, 1]
            kp7 = np.zeros_like(kp0)
            kp7[:, 0] = -kp3[:, 0]
            kp7[:, 1] = kp3[:, 1]

            kp10 = points[0:1, 0:2]
            kp11 = points[n1:n1 + 1, 0:2]
            kp22 = points[n1 + n2:n1 + n2 + 1, 0:2]
            kp14 = points[n1 + n2 + nf:n1 + n2 + nf + 1, 0:2]
            kp8 = points[-1:, 0:2]
            kp9 = points[n1 + 2 * n2 + 2 * nf:n1 + 2 * n2 + 2 * nf + 1, 0:2]
            kp19 = points[n1 + n2 + 2 * nf:n1 + n2 + 2 * nf + 1, 0:2]

            angle_kp8 = np.arctan2(kp8[:, 1], kp8[:, 0])
            angle_kp10 = np.arctan2(kp10[:, 1], kp10[:, 0])
            angle_kp12 = (angle_kp0 + angle_kp8) / 2
            angle_kp13 = pi - angle_kp12
            angle_kp19 = np.arctan2(kp19[:, 1], kp19[:, 0])
            angle_kp22 = pi - angle_kp19
            kp21 = outer_diam / 2 * np.array([cos(angle_kp19), sin(angle_kp19)]).reshape(1, -1)
            kp24 = outer_diam / 2 * np.array([cos(angle_kp22), sin(angle_kp22)]).reshape(1, -1)
            kp13 = ra * np.array([cos(angle_kp13), sin(angle_kp13)]).reshape(1, -1)
            kp12 = ra * np.array([cos(angle_kp12), sin(angle_kp12)]).reshape(1, -1)

            kp15 = np.array([0, outer_diam / 2]).reshape(1, -1)
            r_kp2 = np.sqrt(np.sum(kp2 ** 2))
            t_kp16 = 0.382
            kp16 = (1 - t_kp16) * kp11 + t_kp16 * kp5
            kp18 = (1 - t_kp16) * kp9 + t_kp16 * kp1
            kp17 = np.array([0, r_kp2]).reshape(1, -1)

            kp20 = r_kp2 * np.array([cos(angle_kp19), sin(angle_kp19)]).reshape(1, -1)
            kp23 = r_kp2 * np.array([cos(angle_kp22), sin(angle_kp22)]).reshape(1, -1)

            key_points = np.concatenate(
                [kp0, kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8, kp9, kp10, kp11, kp12, kp13, kp14, kp15, kp16, kp17, kp18,
                 kp19, kp20, kp21, kp22, kp23, kp24],
                axis=0)
            # 构造 edge 与 line
            edge = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [5, 4],
                             [6, 5],
                             [7, 6],
                             [3, 21],
                             [24, 7],
                             [4, 13],
                             [13, 10],
                             [10, 11],
                             [11, 22],
                             [19, 9],
                             [9, 8],
                             [8, 12],
                             [12, 0],
                             [12, 18],
                             [18, 20],
                             [13, 16],
                             [16, 23],
                             [14, 17],
                             [17, 15],
                             [5, 16],
                             [16, 11],
                             [9, 18],
                             [18, 1],
                             [6, 23],
                             [20, 2],
                             [22, 14],
                             [14, 19],
                             [21, 15],
                             [15, 24],
                             [19, 20],
                             [20, 21],
                             [22, 23],
                             [23, 24],
                             [23, 17],
                             [17, 20]
                             ])

            na1 = int(na / 2)
            na2 = na - na1
            delta_kp12_0 = np.linspace(angle_kp12, angle_kp0, na2 + 1)
            delta_kp8_12 = np.linspace(angle_kp8, angle_kp12, na1 + 1)
            delta_kp13_10 = np.linspace(angle_kp13, angle_kp10, na1 + 1)
            delta_kp4_13 = np.linspace(angle_kp4, angle_kp13, na2 + 1)
            delta_kp20_2 = np.linspace(angle_kp19, angle_kp0, na2 + 1)
            delta_kp6_23 = np.linspace(angle_kp4, angle_kp22, na2 + 1)
            delta_kp23_17 = np.linspace(angle_kp22, pi / 2, nf + 1)
            delta_kp17_20 = np.linspace(pi / 2, angle_kp19, nf + 1)
            delta_kp3_21 = delta_kp20_2[::-1]
            delta_kp21_15 = delta_kp17_20[::-1]
            delta_kp15_24 = delta_kp23_17[::-1]
            delta_kp24_7 = delta_kp6_23[::-1]
            line = [
                np.linspace(key_points[edge[0, 0]], key_points[edge[0, 1]], n1 + 1),
                np.linspace(key_points[edge[1, 0]], key_points[edge[1, 1]], n2 + 1),
                np.linspace(key_points[edge[2, 0]], key_points[edge[2, 1]], n3 + 1),
                np.linspace(key_points[edge[3, 0]], key_points[edge[3, 1]], n1 + 1),
                np.linspace(key_points[edge[4, 0]], key_points[edge[4, 1]], n2 + 1),
                np.linspace(key_points[edge[5, 0]], key_points[edge[5, 1]], n3 + 1),
                outer_diam / 2 * np.concatenate([cos(delta_kp3_21), sin(delta_kp3_21)], axis=1),
                outer_diam / 2 * np.concatenate([cos(delta_kp24_7), sin(delta_kp24_7)], axis=1),
                ra * np.concatenate([cos(delta_kp4_13), sin(delta_kp4_13)], axis=1),
                ra * np.concatenate([cos(delta_kp13_10), sin(delta_kp13_10)], axis=1),
                points[0:n1 + 1, :-1],
                points[n1:n1 + n2 + 1, :-1],
                points[n1 + n2 + 2 * nf:n1 + 2 * n2 + 2 * nf + 1, :-1],
                points[n1 + 2 * n2 + 2 * nf:, :-1],
                ra * np.concatenate([cos(delta_kp8_12), sin(delta_kp8_12)], axis=1),
                ra * np.concatenate([cos(delta_kp12_0), sin(delta_kp12_0)], axis=1),
                # np.linspace(key_points[edge[16, 0]], key_points[edge[16, 1]], n1 + 1),
                line_transformer(points[n1 + 2 * n2 + 2 * nf:, :-1][::-1], kp12, kp18),
                line_transformer(points[n1 + n2 + 2 * nf:n1 + 2 * n2 + 2 * nf + 1, :-1][::-1], kp18, kp20),
                # np.linspace(key_points[edge[18, 0]], key_points[edge[18, 1]], n1 + 1),
                line_transformer(points[0:n1 + 1, :-1], kp13, kp16),
                line_transformer(points[n1:n1 + n2 + 1, :-1], kp16, kp23),
                np.linspace(key_points[edge[20, 0]], key_points[edge[20, 1]], na1 + 1),
                np.linspace(key_points[edge[21, 0]], key_points[edge[21, 1]], n3 + 1),
                np.linspace(key_points[edge[22, 0]], key_points[edge[22, 1]], na2 + 1),
                np.linspace(key_points[edge[23, 0]], key_points[edge[23, 1]], na1 + 1),
                np.linspace(key_points[edge[24, 0]], key_points[edge[24, 1]], na1 + 1),
                np.linspace(key_points[edge[25, 0]], key_points[edge[25, 1]], na2 + 1),
                r_kp2 * np.concatenate([cos(delta_kp6_23), sin(delta_kp6_23)], axis=1),
                r_kp2 * np.concatenate([cos(delta_kp20_2), sin(delta_kp20_2)], axis=1),
                rf * np.concatenate([cos(delta_kp23_17), sin(delta_kp23_17)], axis=1),
                rf * np.concatenate([cos(delta_kp17_20), sin(delta_kp17_20)], axis=1),
                outer_diam / 2 * np.concatenate([cos(delta_kp21_15), sin(delta_kp21_15)], axis=1),
                outer_diam / 2 * np.concatenate([cos(delta_kp15_24), sin(delta_kp15_24)], axis=1),
                np.linspace(key_points[edge[32, 0]], key_points[edge[32, 1]], na1 + 1),
                np.linspace(key_points[edge[33, 0]], key_points[edge[33, 1]], n3 + 1),
                np.linspace(key_points[edge[34, 0]], key_points[edge[34, 1]], na1 + 1),
                np.linspace(key_points[edge[35, 0]], key_points[edge[35, 1]], n3 + 1),
                r_kp2 * np.concatenate([cos(delta_kp23_17), sin(delta_kp23_17)], axis=1),
                r_kp2 * np.concatenate([cos(delta_kp17_20), sin(delta_kp17_20)], axis=1)
            ]

            boundary_edge = np.array([0, 1, 2, 6, 30, 31, 7, 5, 4, 3, 8, 9, 10, 11, 28, 29, 12, 13, 14, 15])
            half_edge = subdomain_divider(line, key_points, edge, boundary_edge)

            quad_mesh = QuadrangleMesh.sub_domain_mesh_generator(half_edge, key_points, line)
            cell_domain_tag = quad_mesh.celldata['cell_domain_tag']
            tooth_node = quad_mesh.node
            tooth_cell = quad_mesh.cell
            origin_cell = quad_mesh.cell
            cell_cell_num = len(origin_cell)
            cell_tooth_tag = np.zeros(cell_cell_num * z, dtype=np.int_)

            # 旋转角
            rot_phi = np.linspace(0, 2 * np.pi, z, endpoint=False)

            # 生成完整内齿
            single_node_num = len(tooth_node) - (n1 + n2 + n3 + 1)
            single_cell_num = len(tooth_cell)
            temp_node = np.concatenate(
                [tooth_node[4:len(key_points)], tooth_node[len(key_points) + (n1 + n2 + n3 - 3):]], axis=0)
            temp_node_last = np.concatenate(
                [tooth_node[8:len(key_points)], tooth_node[len(key_points) + 2 * (n1 + n2 + n3 - 3):]], axis=0)
            origin_trans_matrix = np.arange(len(tooth_node))
            trans_matrix = np.arange(len(tooth_node))
            # 左侧齿
            # 处理重复顶点
            trans_matrix[0] = trans_matrix[4]
            trans_matrix[1] = trans_matrix[5]
            trans_matrix[2] = trans_matrix[6]
            trans_matrix[3] = trans_matrix[7]
            # 处理重复边上节点
            trans_matrix[len(key_points):len(key_points) + (n1 - 1)] \
                = trans_matrix[
                  len(key_points) + (n1 + n2 + n3 - 3):len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1)][::-1]
            trans_matrix[len(key_points) + (n1 - 1):len(key_points) + (n1 + n2 - 2)] \
                = trans_matrix[
                  len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1):len(key_points) + 2 * (n1 + n2 - 2) + (n3 - 1)][
                  ::-1]
            trans_matrix[len(key_points) + (n1 + n2 - 2):len(key_points) + (n1 + n2 + n3 - 3)] \
                = trans_matrix[len(key_points) + 2 * (n1 + n2 - 2) + (n3 - 1):len(key_points) + 2 * (n1 + n2 + n3 - 3)][
                  ::-1]

            # 其他节点
            trans_matrix[4:len(key_points)] += single_node_num + (n1 + n2 + n3 - 3)
            trans_matrix[len(key_points) + (n1 + n2 + n3 - 3):] += single_node_num

            rot_matrix = np.array([[np.cos(rot_phi[1]), -np.sin(rot_phi[1])], [np.sin(rot_phi[1]), np.cos(rot_phi[1])]])
            new_node = np.einsum('ij,jn->in', rot_matrix, temp_node.T).T
            new_cell = trans_matrix[origin_cell]
            cell_tooth_tag[cell_cell_num:2 * cell_cell_num] = 1

            tooth_node = np.concatenate([tooth_node, new_node], axis=0)
            tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)

            for i in range(2, z - 1):
                rot_matrix = np.array(
                    [[np.cos(rot_phi[i]), -np.sin(rot_phi[i])], [np.sin(rot_phi[i]), np.cos(rot_phi[i])]])
                new_node = np.einsum('ij,jn->in', rot_matrix, temp_node.T).T

                # 处理重复顶点
                trans_matrix[0] = trans_matrix[4]
                trans_matrix[1] = trans_matrix[5]
                trans_matrix[2] = trans_matrix[6]
                trans_matrix[3] = trans_matrix[7]
                # 处理重复边上节点
                trans_matrix[len(key_points):len(key_points) + (n1 - 1)] \
                    = trans_matrix[
                      len(key_points) + (n1 + n2 + n3 - 3):len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1)][::-1]
                trans_matrix[len(key_points) + (n1 - 1):len(key_points) + (n1 + n2 - 2)] \
                    = trans_matrix[
                      len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1):len(key_points) + 2 * (n1 + n2 - 2) + (
                              n3 - 1)][::-1]
                trans_matrix[len(key_points) + (n1 + n2 - 2):len(key_points) + (n1 + n2 + n3 - 3)] \
                    = trans_matrix[
                      len(key_points) + 2 * (n1 + n2 - 2) + (n3 - 1):len(key_points) + 2 * (n1 + n2 + n3 - 3)][::-1]
                # 其他节点
                trans_matrix[4:len(key_points)] += single_node_num
                trans_matrix[len(key_points) + (n1 + n2 + n3 - 3):] += single_node_num

                new_cell = trans_matrix[origin_cell]
                cell_tooth_tag[i * cell_cell_num:(i + 1) * cell_cell_num] = i
                tooth_node = np.concatenate([tooth_node, new_node], axis=0)
                tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)

            # 右侧齿
            rot_matrix = np.array(
                [[np.cos(rot_phi[-1]), -np.sin(rot_phi[-1])], [np.sin(rot_phi[-1]), np.cos(rot_phi[-1])]])
            new_node = np.einsum('ij,jn->in', rot_matrix, temp_node_last.T).T
            # 处理重复顶点
            trans_matrix[0] = trans_matrix[4]
            trans_matrix[1] = trans_matrix[5]
            trans_matrix[2] = trans_matrix[6]
            trans_matrix[3] = trans_matrix[7]
            trans_matrix[4] = origin_trans_matrix[0]
            trans_matrix[5] = origin_trans_matrix[1]
            trans_matrix[6] = origin_trans_matrix[2]
            trans_matrix[7] = origin_trans_matrix[3]
            # 处理重复边上节点
            trans_matrix[len(key_points):len(key_points) + (n1 - 1)] \
                = trans_matrix[
                  len(key_points) + (n1 + n2 + n3 - 3):len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1)][::-1]
            trans_matrix[len(key_points) + (n1 - 1):len(key_points) + (n1 + n2 - 2)] \
                = trans_matrix[
                  len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1):len(key_points) + 2 * (n1 + n2 - 2) + (n3 - 1)][
                  ::-1]
            trans_matrix[len(key_points) + (n1 + n2 - 2):len(key_points) + (n1 + n2 + n3 - 3)] \
                = trans_matrix[len(key_points) + 2 * (n1 + n2 - 2) + (n3 - 1):len(key_points) + 2 * (n1 + n2 + n3 - 3)][
                  ::-1]
            trans_matrix[len(key_points) + (n1 + n2 + n3 - 3):len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (
                    n3 - 1)] = origin_trans_matrix[len(key_points):len(key_points) + (n1 - 1)][::-1]
            trans_matrix[len(key_points) + 2 * (n1 - 1) + (n2 - 1) + (n3 - 1):len(key_points) + 2 * (n1 + n2 - 2) + (
                    n3 - 1)] = origin_trans_matrix[len(key_points) + (n1 - 1):len(key_points) + (n1 + n2 - 2)][::-1]
            trans_matrix[len(key_points) + 2 * (n1 + n2 - 2) + (n3 - 1):len(key_points) + 2 * (
                    n1 + n2 + n3 - 3)] = origin_trans_matrix[
                                         len(key_points) + (n1 + n2 - 2):len(key_points) + (n1 + n2 + n3 - 3)][::-1]
            # 其他节点
            trans_matrix[8:len(key_points)] += single_node_num - 4
            trans_matrix[len(key_points) + 2 * (n1 + n2 + n3 - 3):] += single_node_num - (n1 + n2 + n3 + 1)

            new_cell = trans_matrix[origin_cell]
            cell_tooth_tag[(z - 1) * cell_cell_num:] = z - 1
            tooth_node = np.concatenate([tooth_node, new_node], axis=0)
            tooth_node = np.dot(tooth_node, np.array([[0, -1], [1, 0]]))
            tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)

            t_mesh = QuadrangleMesh(tooth_node, tooth_cell)
            t_mesh.celldata['cell_domain_tag'] = np.tile(cell_domain_tag, z)
            t_mesh.celldata['cell_tooth_tag'] = cell_tooth_tag

        self.mesh = t_mesh
        return t_mesh


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import json

    # 外齿轮
    # ================================================
    # 参数读取
    with open('data/external_gear_data.json', 'r') as file:
        data = json.load(file)
    m_n = data['mn']  # 法向模数
    z = data['z']  # 齿数
    alpha_n = data['alpha_n']  # 法向压力角
    beta = data['beta']  # 螺旋角
    x_n = data['xn']  # 法向变位系数
    hac = data['hac']  # 齿顶高系数
    cc = data['cc']  # 顶隙系数
    rcc = data['rcc']  # 刀尖圆弧半径
    jn = data['jn']  # 法向侧隙
    n1 = data['n1']  # 渐开线分段数
    n2 = data['n2']  # 过渡曲线分段数
    n3 = data['n3']
    na = data['na']
    nf = data['nf']
    inner_diam = data['inner_diam']  # 轮缘内径
    chamfer_dia = data['chamfer_dia']  # 倒角高度（直径）

    external_gear = ExternalGear(m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, chamfer_dia,
                                 inner_diam)
    quad_mesh = external_gear.generate_mesh()
    external_gear.show_mesh()
    # ==================================================
    # 内齿轮
    # ==================================================
    # 参数读取
    with open('data/internal_gear_data.json', 'r') as file:
        data = json.load(file)
    m_n = data['mn']  # 法向模数
    z = data['z']  # 齿数
    alpha_n = data['alpha_n']  # 法向压力角
    beta = data['beta']  # 螺旋角
    x_n = data['xn']  # 法向变位系数
    hac = data['hac']  # 齿顶高系数
    cc = data['cc']  # 顶隙系数
    rcc = data['rcc']  # 刀尖圆弧半径
    jn = data['jn']  # 法向侧隙
    n1 = data['n1']  # 渐开线分段数
    n2 = data['n2']  # 过渡曲线分段数
    n3 = data['n3']
    na = data['na']
    nf = data['nf']
    outer_diam = data['outer_diam']  # 轮缘内径
    z_cutter = data['z_cutter']
    xn_cutter = data['xn_cutter']

    internal_gear = InternalGear(m_n, z, alpha_n, beta, x_n, hac, cc, rcc, jn, n1, n2, n3, na, nf, outer_diam, z_cutter, xn_cutter)
    q_mesh = internal_gear.generate_mesh()
    internal_gear.show_mesh()