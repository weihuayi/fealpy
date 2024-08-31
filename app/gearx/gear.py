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
        self.d_b = self.d*self.alpha_t
        self.r_b = self.d_b/2

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
        self.d_b = self.d-2*hf
        self.r_b = self.d_b/2
        # 有效齿顶圆
        self.effective_da = self.d_a-self.chamfer_dia
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

        return points, t2

    def generate_mesh(self):
        pass

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

    print(-1)