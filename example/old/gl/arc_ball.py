import numpy as np
from quaternion import Quaternion

class ArcBall():
    def __init__(self, x, y, w, h):
        """
        初始化 Arcball 控制器。
        :param x, y: 鼠标的初始位置。
        :param w, h: 视窗的宽度和高度。
        """
        self.r = np.min([w, h]) / 2  # 半径设为宽度和高度的最小者的一半
        self.center = np.array([w / 2, h / 2])  # 球心设为视窗中心
        self.position = self.project_to_ball(x, y)  # 将初始位置投影到球上

    def update(self, x, y):
        """
        更新由鼠标位置决定的 Arcball 旋转。
        :param x, y: 当前鼠标位置。
        :return: 表示旋转的四元数。
        """
        p0 = self.position
        p1 = self.project_to_ball(x, y)
        data = np.zeros(4, dtype=np.float_)
        data[0:3] = np.cross(p0, p1)
        data[-1] = np.dot(p0, p1)
        q = Quaternion(data)
        self.position = p1
        return q

    def project_to_ball(self, x, y):
        """
        将屏幕坐标 (x, y) 投影到虚拟球面上。
        :param x, y: 屏幕坐标。
        :return: 球面上的坐标。
        """
        p = np.array([x - self.center[0], y - self.center[1], 0], dtype=np.float_)
        p /= self.r
        r_squared = np.dot(p[:2], p[:2])
        if r_squared > 1.0:
            p[:2] /= np.sqrt(r_squared)  # 投影到球的边界上
        else:
            p[2] = np.sqrt(1 - r_squared)  # 计算z坐标
        return p
