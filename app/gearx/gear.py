from abc import ABC, abstractmethod
import numpy as np
from numpy import sin, cos, tan, pi, arctan, arctan2, radians

from scipy.optimize import fsolve
from fealpy.experimental.mesh.quadrangle_mesh import QuadrangleMesh


class Gear(ABC):
    def __init__(self, m_n, z, alpha_n, beta, x_n, hac, cc, rco, jn, n1, n2, n3, na, nf, chamfer_dia):
        """

        @param m_n: 法向模数
        @param z: 齿数
        @param alpha_n: 法向压力角
        @param beta: 螺旋角
        @param x_n: 法向变位系数
        @param hac: 齿顶高系数
        @param cc: 顶隙系数
        @param rco: 刀尖圆弧半径系数
        @param jn: 法向侧隙
        @param n1: 渐开线分段数
        @param n2: 过渡曲线分段数
        @param n3: 齿轮内部分段书
        @param na: 齿顶分段数
        @param nf: 齿根圆部分分段数（一侧，非最大圆角时）
        @param chamfer_dia: 倒角高度（直径方向）
        """
        if not isinstance(z, int) or (isinstance(z, float) and not z.is_integer()):
            raise TypeError(f'The provided value {z} is not an integer or cannot be safely converted to an integer.')
        self.m_n = m_n
        self.z = z
        self.alpha_n = alpha_n if alpha_n < 2*pi else radians(alpha_n)
        self.beta = beta if beta < 2*pi else radians(beta)
        self.x_n = x_n
        self.hac = hac
        self.cc = cc
        self.rco = rco
        self.jn = jn
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.na = na
        self.nf = nf
        self.chamfer_dia = chamfer_dia
        self.mesh = None

        # 端面变位系数
        self.x_t = self.x_n/cos(self.beta)
        # 端面压力角
        self.alpha_t = arctan(tan(self.alpha_n)/cos(self.beta))
        # 端面模数
        self.m_t = self.m_n/cos(self.beta)
        # 分度圆直径与半径
        self.d = self.m_t*self.z
        self.r = self.d/2
        # 基圆（base circle）直径与半径
        self.d_f = self.d * self.alpha_t
        self.r_f = self.d_f / 2

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

    @abstractmethod
    def optimize_parameters(self):
        pass

    def show_mesh(self, save_path=None):
        if self.mesh is None:
            raise AssertionError('The mesh is not yet created.')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(38, 20))
        self.mesh.add_plot(ax, linewidth=0.1)
        if save_path is not None:
            plt.savefig('./image/全齿网格生成_实际齿顶圆_24_8_30.png', dpi=600, bbox_inches='tight')
        plt.show()


class ExternalGear(Gear):
    def __init__(self, m_n, z, alpha_n, beta, x_n, hac, cc, rco, jn, n1, n2, n3, na, nf, chamfer_dia, inner_diam):
        """

        @param m_n: 法向模数
        @param z: 齿数
        @param alpha_n: 法向压力角
        @param beta: 螺旋角
        @param x_n: 法向变位系数
        @param hac: 齿顶高系数
        @param cc: 顶隙系数
        @param rco: 刀尖圆弧半径
        @param jn: 法向侧隙
        @param n1: 渐开线分段数
        @param n2: 过渡曲线分段数
        @param n3: 齿轮内部分段书
        @param na: 齿顶分段数
        @param nf: 齿根圆部分分段数（一侧，非最大圆角时）
        @param chamfer_dia: 倒角高度（直径方向）
        @param inner_diam: 轮缘内径
        """
        super().__init__(m_n, z, alpha_n, beta, x_n, hac, cc, rco, jn, n1, n2, n3, na, nf, chamfer_dia)
        self.inner_diam = inner_diam
        # 齿顶圆直径与半径
        ha = self.m_n*(self.hac+self.x_t)  # 齿顶高
        self.d_a = self.d+2*ha
        self.r_a = self.d_a/2
        # 齿根圆直径与半径
        hf = self.m_n*(self.hac+self.cc-self.x_t)
        self.d_f = self.d - 2 * hf
        self.r_f = self.d_f / 2
        # 有效齿顶圆
        self.effective_da = self.d_a-self.chamfer_dia
        self.effective_ra = self.effective_da/2
        # 刀具齿顶高与刀尖圆弧半径
        self.ha_cutter = (self.hac + self.cc) * self.m_n
        self.r_cutter = self.m_n * self.rco

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
              (np.cos(phi) * (k + t * np.cos(np.pi / 2 - alpha_t))) / np.cos(beta))

        yt = (r * np.cos(phi) + phi * r * np.sin(phi) +
              t * np.cos(phi) * np.sin(np.pi / 2 - alpha_t) -
              (np.sin(phi) * (k + t * np.cos(np.pi / 2 - alpha_t))) / np.cos(beta))

        return xt, yt

    def get_tip_intersection_points(self, t):

        xt, yt = self.get_involute_points(t)
        return np.sqrt(xt ** 2 + yt ** 2)

    def get_transition_points(self, t):
        r = self.r
        r_cutter = self.r_cutter  # 刀尖圆弧半径
        ha_cutter = self.ha_cutter  # 刀具齿顶高
        alpha_t = self.alpha_t
        beta = self.beta

        # 刀尖圆弧 y 坐标
        x0 = -np.pi * self.m_n / 2 + (np.pi * self.m_n / 4 - ha_cutter * np.tan(alpha_t) - r_cutter * np.tan(0.25 * np.pi - 0.5 * alpha_t))
        # 刀尖圆弧 y 坐标
        y0 = -(ha_cutter - r_cutter) + self.m_n * self.x_t

        phi = (x0 * np.sin(t) + r_cutter * np.cos(t) * np.sin(t) - y0 * np.cos(beta) ** 2 * np.cos(t) - r_cutter * np.cos(
            beta) ** 2 * np.cos(t) * np.sin(t)) / (r * np.cos(beta) * np.sin(t))

        xt = (r * np.sin(phi) + np.sin(phi) * (y0 + r_cutter * np.sin(t)) - phi * r * np.cos(phi) +
              (np.cos(phi) * (x0 + r_cutter * np.cos(t))) / np.cos(beta))

        yt = (r * np.cos(phi) + np.cos(phi) * (y0 + r_cutter * np.sin(t)) + phi * r * np.sin(phi) -
              (np.sin(phi) * (x0 + r_cutter * np.cos(t))) / np.cos(beta))

        return xt, yt
    
    def get_profile_points(self):
        n1 = self.n1
        n2 = self.n2
        mn = self.m_n
        alpha_t = self.alpha_t
        beta = self.beta
        z = self.z
        x = self.x_t

        xt1 = np.zeros((n1 + n2 + 1) * 2)
        yt1 = np.zeros((n1 + n2 + 1) * 2)
        points = np.zeros(((n1 + n2 + 1) * 2, 3))

        mt = self.m_t

        d = self.d
        effective_da = self.effective_da
        ha_cutter = self.ha_cutter  # 刀具齿顶高
        r_cutter = self.r_cutter  # 刀尖圆弧半径

        t1 = (mn * x - (ha_cutter - r_cutter + r_cutter * sin(alpha_t))) / cos(alpha_t)

        def involutecross(t2):
            return self.get_tip_intersection_points(t2) - (0.5 * effective_da)

        t2 = fsolve(involutecross, mn)[0]  # 求解渐开线与齿顶圆的交点

        t3 = 2 * np.pi - alpha_t
        t4 = 1.5 * np.pi
        width2 = t3 - t4
        t = t4 - width2 / n2

        for i in range(n2 + 1):
            t += width2 / n2
            xt1[i], yt1[i] = self.get_transition_points(t)

        width1 = t2 - t1
        t = t1

        for i in range(n2 + 1, n1 + n2 + 1):
            t += width1 / n1
            xt1[i], yt1[i] = self.get_involute_points(t)

        for i in range(n1 + n2 + 1):
            xt1[n1 + n2 + 1 + i] = -xt1[i]
            yt1[n1 + n2 + 1 + i] = yt1[i]

        for i in range((n1 + n2 + 1) * 2):
            points[i, 0] = xt1[i]
            points[i, 1] = yt1[i]
            points[i, 2] = 0

        return points

    def generate_mesh(self):
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3
        nf = self.nf
        na = self.na
        rf = self.r_f
        ra = self.effective_ra
        # 获取齿廓与过渡曲线点列
        points = self.get_profile_points()
        # 齿顶弧线，逆时针，角度参数 t_aa > t_a
        t_a = arctan2(points[-1, 1], points[-1, 0])
        t_aa = pi - t_a

        # 齿根部分
        t_ff = arctan2(points[0, 1], points[0, 0])
        t_f = pi - t_ff
        r_inner = self.inner_diam / 2
        theta = np.linspace(t_f, t_ff, 100)
        x_f = r_inner * cos(theta)
        y_f = r_inner * sin(theta)
        
        # 构造关键点
        kp_1 = points[n1 + n2 + 1]
        kp_4 = points[0]

        kp_2 = points[2 * n1 + n2 + 1]
        kp_5 = points[n1]

        kp_3 = points[-1]
        kp_6 = points[n1 + n2]

        kp_0 = np.array([x_f[0], y_f[0], 0])
        kp_11 = np.array([x_f[-1], y_f[-1], 0])

        kp_10 = np.array([0, r_inner, 0])
        kp_9 = np.array([0, ra, 0])

        # 单侧弧长与点数，计算中轴上点参数
        distance = np.sqrt(np.sum(np.diff(points[:n1 + n2 + 1], axis=0) ** 2, axis=1))
        length2 = np.sum(distance[:n1])
        length3 = np.sum(distance[n1:n1 + n2])
        length1 = np.sqrt(np.sum((kp_4 - kp_11) ** 2))

        n_total = n1 * 1.236 + n2 * 0.618 + n3
        length2_n = length2 * (n1 * 1.236 / n_total)
        length3_n = length3 * (n2 * 0.618 / n_total)
        length1_n = length1 * (n3 / n_total)
        length_total_n = length1_n + length2_n + length3_n

        t_2 = length2_n / length_total_n
        t_1 = length1_n / length_total_n

        kp_7 = np.array([0, r_inner + (ra - r_inner) * t_1, 0])
        kp_8 = np.array([0, r_inner + (ra - r_inner) * (t_1 + t_2), 0])

        # 旋转角
        rot_phi = np.linspace(0, 2 * np.pi, z, endpoint=False)

        # 齿根圆弧上点计算
        rot_kp_1 = np.zeros(2)
        rot_kp_1[0] = np.cos(rot_phi[1]) * kp_1[0] - np.sin(rot_phi[1]) * kp_1[1]
        rot_kp_1[1] = np.sin(rot_phi[1]) * kp_1[0] + np.cos(rot_phi[1]) * kp_1[1]
        angle0 = np.arctan2(kp_1[1], kp_1[0])
        angle1 = np.arctan2(rot_kp_1[1], rot_kp_1[0])
        angle2 = np.arctan2(kp_4[1], kp_4[0])
        delta_angle = abs(angle1 - angle2)

        # TODO: 改用齿根圆角是否超过最大圆角进行判断与分类
        # 两侧过渡曲线之间相连，齿槽底面为一条直线，宽度为 0
        if delta_angle < 1e-12:
            key_points = np.array([kp_0, kp_1, kp_2, kp_3, kp_4, kp_5, kp_6, kp_7, kp_8, kp_9, kp_10, kp_11])

            edge = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [10, 7],
                             [7, 8],
                             [8, 9],
                             [11, 4],
                             [4, 5],
                             [5, 6],
                             [10, 0],
                             [7, 1],
                             [8, 2],
                             [9, 3],
                             [11, 10],
                             [4, 7],
                             [5, 8],
                             [6, 9]])

            # 构建子区域半边数据结构
            half_edge = np.zeros((len(edge) * 2, 5), dtype=np.int64)
            half_edge[::2, 0] = edge[:, 1]
            half_edge[1::2, 0] = edge[:, 0]
            half_edge[::2, 4] = 2 * np.arange(len(edge)) + 1
            half_edge[1::2, 4] = 2 * np.arange(len(edge))
            half_edge[np.array([0, 1, 2]) * 2, 1] = np.array([0, 1, 2])
            half_edge[np.array([0, 1, 2]) * 2 + 1, 1] = -1
            half_edge[np.array([3, 4, 5]) * 2, 1] = np.array([3, 4, 5])
            half_edge[np.array([3, 4, 5]) * 2 + 1, 1] = np.array([0, 1, 2])
            half_edge[np.array([6, 7, 8]) * 2, 1] = -1
            half_edge[np.array([6, 7, 8]) * 2 + 1, 1] = np.array([3, 4, 5])
            half_edge[np.array([10, 11, 14, 15]) * 2, 1] = np.array([1, 2, 4, 5])
            half_edge[np.array([10, 11, 14, 15]) * 2 + 1, 1] = np.array([0, 1, 3, 4])
            half_edge[np.array([9, 13]) * 2, 1] = np.array([0, 3])
            half_edge[np.array([9, 13]) * 2 + 1, 1] = -1
            half_edge[np.array([12, 16]) * 2, 1] = -1
            half_edge[np.array([12, 16]) * 2 + 1, 1] = np.array([2, 5])

            half_edge[::2, 2] = np.array([21, 23, 25, 29, 31, 33, 14, 16, 32, 0, 2, 4, 5, 6, 8, 10, 24])
            half_edge[1::2, 2] = np.array([19, 1, 3, 18, 20, 22, 26, 28, 30, 27, 7, 9, 11, 12, 13, 15, 17])

            half_edge[::2, 3] = np.array([18, 20, 22, 26, 28, 30, 27, 12, 14, 7, 9, 11, 32, 13, 15, 17, 16])
            half_edge[1::2, 3] = np.array([3, 5, 24, 21, 23, 25, 29, 31, 33, 1, 0, 2, 4, 19, 6, 8, 10])

            theta_f = np.linspace(np.pi / 2, t_f, na + 1)
            theta_ff = np.linspace(t_ff, np.pi / 2, na + 1)
            theta_a = np.linspace(np.pi / 2, t_a, na + 1)
            theta_aa = np.linspace(t_aa, np.pi / 2, na + 1)
            line = [
                np.linspace(kp_0[..., :-1], kp_1[..., :-1], n3 + 1),
                points[n1 + n2 + 1:2 * n1 + n2 + 2, :-1],
                points[2 * n1 + n2 + 1:2 * n1 + 2 * n2 + 2,
                :-1],
                np.linspace(kp_10[..., :-1], kp_7[..., :-1], n3 + 1),
                np.linspace(kp_7[..., :-1], kp_8[..., :-1], n1 + 1),
                np.linspace(kp_8[..., :-1], kp_9[..., :-1], n2 + 1),
                np.linspace(kp_11[..., :-1], kp_4[..., :-1], n3 + 1),
                points[:n1 + 1, :-1],
                points[n1:n1 + n2 + 1, :-1],
                np.concatenate([r_inner * np.cos(theta_f)[:, None], r_inner * np.sin(theta_f)[:, None]], axis=1),
                np.linspace(kp_7[..., :-1], kp_1[..., :-1], na + 1),
                np.linspace(kp_8[..., :-1], kp_2[..., :-1], na + 1),
                np.concatenate([ra * np.cos(theta_a)[:, None], ra * np.sin(theta_a)[:, None]], axis=1),
                np.concatenate([r_inner * np.cos(theta_ff)[:, None], r_inner * np.sin(theta_ff)[:, None]], axis=1),
                np.linspace(kp_4[..., :-1], kp_7[..., :-1], na + 1),
                np.linspace(kp_5[..., :-1], kp_8[..., :-1], na + 1),
                np.concatenate([ra * np.cos(theta_aa)[:, None], ra * np.sin(theta_aa)[:, None]], axis=1)
            ]

            quad_mesh = QuadrangleMesh.sub_domain_mesh_generator(half_edge, key_points[:, :-1], line)
            tooth_node = quad_mesh.node
            tooth_cell = quad_mesh.cell

            single_node_num = len(tooth_node) - (n3 + 1)
            single_cell_num = len(tooth_cell)
            temp_node = np.concatenate([tooth_node[2:len(key_points)], tooth_node[(len(key_points) + (n3 - 1)):]],
                                       axis=0)
            # 左侧齿
            trans_matrix = np.arange(len(tooth_node))
            # 处理重复顶点
            trans_matrix[0] = trans_matrix[11]
            trans_matrix[1] = trans_matrix[4]
            # 处理重复边上节点
            trans_matrix[len(key_points):len(key_points) + n3 - 1] = trans_matrix[len(key_points) + 2 * (
                    n3 + n1 + n2 - 3):len(key_points) + 2 * (
                    n3 + n1 + n2 - 3) + n3 - 1]
            # 其他节点
            trans_matrix[2:len(key_points)] += single_node_num + n3 - 1
            trans_matrix[len(key_points) + n3 - 1:] += single_node_num

            rot_matrix = np.array([[np.cos(rot_phi[1]), -np.sin(rot_phi[1])], [np.sin(rot_phi[1]), np.cos(rot_phi[1])]])
            new_node = np.einsum('ij,jn->in', rot_matrix, temp_node.T).T
            new_cell = trans_matrix[tooth_cell]

            tooth_node = np.concatenate([tooth_node, new_node], axis=0)
            tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)
            # 中间齿
            for i in range(2, z - 1):
                rot_matrix = np.array(
                    [[np.cos(rot_phi[i]), -np.sin(rot_phi[i])], [np.sin(rot_phi[i]), np.cos(rot_phi[i])]])
                new_node = np.einsum('ij,jn->in', rot_matrix, temp_node.T).T
                # 处理重复顶点
                trans_matrix[0] = trans_matrix[11]
                trans_matrix[1] = trans_matrix[4]
                # 处理重复边上节点
                trans_matrix[len(key_points):len(key_points) + n3 - 1] = trans_matrix[len(key_points) + 2 * (
                        n3 + n1 + n2 - 3):len(key_points) + 2 * (
                        n3 + n1 + n2 - 3) + n3 - 1]
                # 其他节点
                trans_matrix[0:12] += single_node_num
                trans_matrix[14:len(key_points) + edge_node_num - (4 * (nf - 1) + 2 * (n3 - 1))] += single_node_num
                trans_matrix[len(key_points) + edge_node_num - (4 * (nf - 1) + (n3 - 1)):] += single_node_num
                # 新单元映射与拼接
                new_cell = trans_matrix[origin_cell]
                tooth_node = np.concatenate([tooth_node, new_node], axis=0)
                tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)

            t_mesh = QuadrangleMesh(tooth_node, tooth_cell)
        else:
            # 计算边内部点数
            edge_node_num = (na - 1) * 8 + (n2 - 1) * 3 + (n1 - 1) * 3 + (n3 - 1) * 5 + (
                    nf - 1) * 4
            # 构造剩余关键点
            kp_12_angle = angle0 - delta_angle / 2
            kp_14_angle = angle2 + delta_angle / 2
            kp_12 = np.array([r_inner * np.cos(kp_12_angle), r_inner * np.sin(kp_12_angle), 0])
            kp_13 = np.array([rf * np.cos(kp_12_angle), rf * np.sin(kp_12_angle), 0])
            kp_14 = np.array([r_inner * np.cos(kp_14_angle), r_inner * np.sin(kp_14_angle), 0])
            kp_15 = np.array([rf * np.cos(kp_14_angle), rf * np.sin(kp_14_angle), 0])
            key_points = np.array(
                [kp_0, kp_1, kp_2, kp_3, kp_4, kp_5, kp_6, kp_7, kp_8, kp_9, kp_10, kp_11, kp_12, kp_13, kp_14, kp_15])
            # 构造半边数据结构所用分区边
            edge = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [10, 7],
                             [7, 8],
                             [8, 9],
                             [11, 4],
                             [4, 5],
                             [5, 6],
                             [10, 0],
                             [7, 1],
                             [8, 2],
                             [9, 3],
                             [11, 10],
                             [4, 7],
                             [5, 8],
                             [6, 9],
                             [12, 13],
                             [14, 15],
                             [0, 12],
                             [1, 13],
                             [14, 11],
                             [15, 4]])
            # 构建子区域半边数据结构
            half_edge = np.zeros((len(edge) * 2, 5), dtype=np.int64)
            half_edge[::2, 0] = edge[:, 1]
            half_edge[1::2, 0] = edge[:, 0]

            half_edge[::2, 4] = 2 * np.arange(len(edge)) + 1
            half_edge[1::2, 4] = 2 * np.arange(len(edge))

            half_edge[np.array([0, 1, 2]) * 2, 1] = np.array([0, 1, 2])
            half_edge[np.array([1, 2]) * 2 + 1, 1] = -1
            half_edge[0 * 2 + 1, 1] = 6
            half_edge[np.array([3, 4, 5]) * 2, 1] = np.array([3, 4, 5])
            half_edge[np.array([3, 4, 5]) * 2 + 1, 1] = np.array([0, 1, 2])
            half_edge[np.array([7, 8]) * 2, 1] = -1
            half_edge[6 * 2, 1] = 7
            half_edge[np.array([6, 7, 8]) * 2 + 1, 1] = np.array([3, 4, 5])
            half_edge[np.array([10, 11, 14, 15]) * 2, 1] = np.array([1, 2, 4, 5])
            half_edge[np.array([10, 11, 14, 15]) * 2 + 1, 1] = np.array([0, 1, 3, 4])
            half_edge[np.array([9, 13]) * 2, 1] = np.array([0, 3])
            half_edge[np.array([9, 13]) * 2 + 1, 1] = -1
            half_edge[np.array([12, 16]) * 2, 1] = -1
            half_edge[np.array([12, 16]) * 2 + 1, 1] = np.array([2, 5])
            half_edge[17 * 2, 1] = 6
            half_edge[17 * 2 + 1, 1] = -1
            half_edge[18 * 2, 1] = -1
            half_edge[18 * 2 + 1, 1] = 7
            half_edge[np.array([19, 21]) * 2, 1] = np.array([6, 7])
            half_edge[np.array([19, 21]) * 2 + 1, 1] = -1
            half_edge[np.array([20, 22]) * 2, 1] = -1
            half_edge[np.array([20, 22]) * 2 + 1, 1] = np.array([6, 7])

            half_edge[::2, 2] = np.array(
                [21, 23, 25, 29, 31, 33, 45, 16, 32, 0, 2, 4, 5, 6, 8, 10, 24, 41, 44, 34, 35, 12, 14])
            half_edge[1::2, 2] = np.array(
                [38, 40, 3, 18, 20, 22, 26, 28, 30, 27, 7, 9, 11, 43, 13, 15, 17, 39, 42, 19, 1, 36, 37])

            half_edge[::2, 3] = np.array(
                [18, 20, 22, 26, 28, 30, 42, 44, 14, 7, 9, 11, 32, 13, 15, 17, 16, 38, 43, 1, 3, 37, 36])
            half_edge[1::2, 3] = np.array(
                [41, 5, 24, 21, 23, 25, 29, 31, 33, 39, 0, 2, 4, 19, 6, 8, 10, 40, 45, 35, 34, 27, 12])
            # 构建半边数据结构所用边（由点列构成的边）
            theta_f = np.linspace(np.pi / 2, t_f, na + 1)
            theta_ff = np.linspace(t_ff, np.pi / 2, na + 1)
            theta_a = np.linspace(np.pi / 2, t_a, na + 1)
            theta_aa = np.linspace(t_aa, np.pi / 2, na + 1)
            theta_b1 = np.linspace(kp_12_angle, t_f, nf + 1)
            theta_b2 = np.linspace(t_ff, kp_14_angle, nf + 1)
            line = [
                np.linspace(kp_0[..., :-1], kp_1[..., :-1], n3 + 1),
                points[n1 + n2 + 1:2 * n1 + n2 + 2, :-1],
                points[2 * n1 + n2 + 1:2 * n1 + 2 * n2 + 2,
                :-1],
                np.linspace(kp_10[..., :-1], kp_7[..., :-1], n3 + 1),
                np.linspace(kp_7[..., :-1], kp_8[..., :-1], n1 + 1),
                np.linspace(kp_8[..., :-1], kp_9[..., :-1], n2 + 1),
                np.linspace(kp_11[..., :-1], kp_4[..., :-1], n3 + 1),
                points[:n1 + 1, :-1],
                points[n1:n1 + n2 + 1, :-1],
                np.concatenate([r_inner * np.cos(theta_f)[:, None], r_inner * np.sin(theta_f)[:, None]], axis=1),
                np.linspace(kp_7[..., :-1], kp_1[..., :-1], na + 1),
                np.linspace(kp_8[..., :-1], kp_2[..., :-1], na + 1),
                np.concatenate([ra * np.cos(theta_a)[:, None], ra * np.sin(theta_a)[:, None]], axis=1),
                np.concatenate([r_inner * np.cos(theta_ff)[:, None], r_inner * np.sin(theta_ff)[:, None]], axis=1),
                np.linspace(kp_4[..., :-1], kp_7[..., :-1], na + 1),
                np.linspace(kp_5[..., :-1], kp_8[..., :-1], na + 1),
                np.concatenate([ra * np.cos(theta_aa)[:, None], ra * np.sin(theta_aa)[:, None]], axis=1),
                np.linspace(kp_12[..., :-1], kp_13[..., :-1], n3 + 1),
                np.linspace(kp_14[..., :-1], kp_15[..., :-1], n3 + 1),
                np.concatenate([r_inner * np.cos(theta_b1)[:, None], r_inner * np.sin(theta_b1)[:, None]], axis=1),
                np.concatenate([rf * np.cos(theta_b1)[:, None], rf * np.sin(theta_b1)[:, None]], axis=1),
                np.concatenate([r_inner * np.cos(theta_b2)[:, None], r_inner * np.sin(theta_b2)[:, None]], axis=1),
                np.concatenate([rf * np.cos(theta_b2)[:, None], rf * np.sin(theta_b2)[:, None]], axis=1)
            ]
            # 单齿网格及其节点与单元
            quad_mesh = QuadrangleMesh.sub_domain_mesh_generator(half_edge, key_points[:, :-1], line)
            tooth_node = quad_mesh.node
            tooth_cell = quad_mesh.cell
            origin_cell = quad_mesh.cell
            # 旋转构建剩余点与单元，并依次拼接
            single_node_num = len(tooth_node) - (n3 + 1)
            temp_node = np.concatenate(
                [tooth_node[:12], tooth_node[14:len(key_points) + edge_node_num - (4 * (nf - 1) + 2 * (n3 - 1))],
                 tooth_node[len(key_points) + edge_node_num - (4 * (nf - 1) + (n3 - 1)):]], axis=0)
            # 最后一个齿的节点，需要特殊处理
            temp_node_last = np.concatenate(
                [tooth_node[:12], tooth_node[16:len(key_points) + edge_node_num - (4 * (nf - 1) + 2 * (n3 - 1))],
                 tooth_node[len(key_points) + edge_node_num - (4 * (nf - 1)):]], axis=0)
            # 辅助所用的节点映射，将新节点编号按照初始单元节点排列
            origin_trans_matrix = np.arange(len(tooth_node))
            trans_matrix = np.arange(len(tooth_node))
            # 左侧齿
            # 处理重复顶点
            trans_matrix[12] = trans_matrix[14]
            trans_matrix[13] = trans_matrix[15]
            # 处理重复边上节点
            trans_matrix[len(key_points) + edge_node_num - (4 * (nf - 1) + 2 * (n3 - 1))
                         :len(key_points) + edge_node_num - (4 * (nf - 1) + (n3 - 1))] \
                = trans_matrix[len(key_points) + edge_node_num - (4 * (nf - 1) + (n3 - 1))
                               :len(key_points) + edge_node_num - (4 * (nf - 1))]
            # 其他节点
            trans_matrix[0:12] += single_node_num + (n3 - 1) + 2
            trans_matrix[14:len(key_points) + edge_node_num - (4 * (nf - 1) + 2 * (n3 - 1))] += single_node_num + (
                        n3 - 1)
            trans_matrix[len(key_points) + edge_node_num - (4 * (nf - 1) + (n3 - 1)):] += single_node_num
            # 计算新节点与单元
            rot_matrix = np.array([[np.cos(rot_phi[1]), -np.sin(rot_phi[1])], [np.sin(rot_phi[1]), np.cos(rot_phi[1])]])
            new_node = np.einsum('ij,jn->in', rot_matrix, temp_node.T).T
            new_cell = trans_matrix[origin_cell]
            # 拼接
            tooth_node = np.concatenate([tooth_node, new_node], axis=0)
            tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)
            # 中间齿
            for i in range(2, z - 1):
                rot_matrix = np.array(
                    [[np.cos(rot_phi[i]), -np.sin(rot_phi[i])], [np.sin(rot_phi[i]), np.cos(rot_phi[i])]])
                new_node = np.einsum('ij,jn->in', rot_matrix, temp_node.T).T
                # 处理重复顶点
                trans_matrix[12] = trans_matrix[14]
                trans_matrix[13] = trans_matrix[15]
                # 处理重复边上节点
                trans_matrix[len(key_points) + edge_node_num - (4 * (nf - 1) + 2 * (n3 - 1))
                             :len(key_points) + edge_node_num - (4 * (nf - 1) + (n3 - 1))] \
                    = trans_matrix[len(key_points) + edge_node_num - (4 * (nf - 1) + (n3 - 1))
                                   :len(key_points) + edge_node_num - (4 * (nf - 1))]
                # 其他节点
                trans_matrix[0:12] += single_node_num
                trans_matrix[14:len(key_points) + edge_node_num - (4 * (nf - 1) + 2 * (n3 - 1))] += single_node_num
                trans_matrix[len(key_points) + edge_node_num - (4 * (nf - 1) + (n3 - 1)):] += single_node_num
                # 新单元映射与拼接
                new_cell = trans_matrix[origin_cell]
                tooth_node = np.concatenate([tooth_node, new_node], axis=0)
                tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)
            # 右侧齿
            rot_matrix = np.array(
                [[np.cos(rot_phi[-1]), -np.sin(rot_phi[-1])], [np.sin(rot_phi[-1]), np.cos(rot_phi[-1])]])
            new_node = np.einsum('ij,jn->in', rot_matrix, temp_node_last.T).T
            # 处理重复顶点
            trans_matrix[12] = trans_matrix[14]
            trans_matrix[13] = trans_matrix[15]
            trans_matrix[14] = origin_trans_matrix[12]
            trans_matrix[15] = origin_trans_matrix[13]
            # 处理重复边上节点
            trans_matrix[len(key_points) + edge_node_num - (4 * (nf - 1) + 2 * (n3 - 1))
                         :len(key_points) + edge_node_num - (4 * (nf - 1) + (n3 - 1))] \
                = trans_matrix[len(key_points) + edge_node_num - (4 * (nf - 1) + (n3 - 1))
                               :len(key_points) + edge_node_num - (4 * (nf - 1))]
            trans_matrix[len(key_points) + edge_node_num - (4 * (nf - 1) + (n3 - 1))
                         :len(key_points) + edge_node_num - (4 * (nf - 1))] \
                = origin_trans_matrix[len(key_points) + edge_node_num - (4 * (nf - 1) + 2 * (n3 - 1))
                                      :len(key_points) + edge_node_num - (4 * (nf - 1) + (n3 - 1))]
            # 其他节点
            trans_matrix[0:12] += single_node_num
            trans_matrix[16:len(key_points) + edge_node_num - (4 * (nf - 1) + 2 * (n3 - 1))] += single_node_num - 2
            trans_matrix[len(key_points) + edge_node_num - (4 * (nf - 1)):] += single_node_num - (n3 - 1) - 2
            # 新单元映射与拼接
            new_cell = trans_matrix[origin_cell]
            tooth_node = np.concatenate([tooth_node, new_node], axis=0)
            tooth_cell = np.concatenate([tooth_cell, new_cell], axis=0)
            # 最终网格
            t_mesh = QuadrangleMesh(tooth_node, tooth_cell)

        self.mesh = t_mesh
        return t_mesh


    def optimize_parameters(self):
        pass



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import json

    # 参数读取
    with open('./external_gear_data.json', 'r') as file:
        data = json.load(file)
    m_n = data['mn']  # 法向模数
    z = data['z']  # 齿数
    alpha_n = data['alpha_n']  # 法向压力角
    beta = data['beta']  # 螺旋角
    x_n = data['x']  # 法向变位系数
    hac = data['hac']  # 齿顶高系数
    cc = data['cc']  # 顶隙系数
    rco = data['rco']  # 刀尖圆弧半径
    jn = data['jn']  # 法向侧隙
    n1 = data['involute_section']  # 渐开线分段数
    n2 = data['transition_section']  # 过渡曲线分段数
    n3 = data['n3']
    na = data['na']
    nf = data['nf']
    inner_diam = data['inner_diam']  # 轮缘内径
    chamfer_dia = data['chamfer_dia']  # 倒角高度（直径）

    external_gear = ExternalGear(m_n, z, alpha_n, beta, x_n, hac, cc, rco, jn, n1, n2, n3, na, nf, chamfer_dia, inner_diam)
    quad_mesh = external_gear.generate_mesh()
    external_gear.show_mesh()

