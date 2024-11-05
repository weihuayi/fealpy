import numpy as np

class Quaternion:
    def __init__(self, arr):
        self.data = np.array(arr, dtype=float)

    def norm(self):
        """计算四元数的模（长度）。"""
        return np.linalg.norm(self.data)

    def normalize(self):
        """将四元数归一化为单位四元数。"""
        norm = self.norm()
        if norm != 0:
            self.data /= norm

    def __xor__(self, other):
        """重载异或运算符用于计算点积。"""
        return np.dot(self.data, other.data)

    def __itruediv__(self, m):
        """重载 /= 运算符。"""
        self.data /= m
        return self

    def __mul__(self, other):
        """重载 * 运算符用于四元数乘法。"""
        w0, x0, y0, z0 = self.data
        w1, x1, y1, z1 = other.data
        w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        return Quaternion([w, x, y, z])

    def convert_to_opengl_matrix(self):
        """将单位四元数转换为OpenGL旋转矩阵。"""
        q = self.data
        n = np.dot(q, q)
        if n < np.finfo(float).eps:
            return np.identity(4, dtype=float)

        q *= 2.0 / n
        x, y, z, w = q
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        matrix = np.array([
            [1.0-yy-zz, xy+wz, xz-wy, 0.0],
            [xy-wz, 1.0-xx-zz, yz+wx, 0.0],
            [xz+wy, yz-wx, 1.0-xx-yy, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)

        return matrix.flatten()