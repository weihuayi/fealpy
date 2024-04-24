import numpy as np

class Quaternion():
    def __init__(self, arr):
        self.data = arr

    def norm(self):
        return np.linalg.norm(self.data) 

    def normlize(self):
        self.data = self.data/self.norm()

    def __xor__(self, other):
        """
        操作符 ^ 重载
        """
        return self.data@other.data

    def __itruediv(self, m):
        self.data /= m
        return self

    def __mul__(self, other):
        q0 = self.data[:-1]
        q1 = other.data[:-1]
        w0 = self.data[-1]
        w1 = other.data[-1]

        w = w0*w1 - q0@q1
        q = w0*q1+w1*q0+np.cross(q0, q1)
        re = Quaternion(np.append(q, w))
        return re 

    def convert_to_opengl_matrix(self):
        """
        q 是一个长度为 4 的数组, 代表一个四元数 q[0]i + q[1]j + q[2]k + q[3], 
        将 q 转换为一个旋转矩阵 m, m 是一个长度为 16 的数组代表一个 4*4 的矩阵.
        """
        m = np.zeros(16, dtype=np.float_)
        q = self.data
        l = q@q;
        s = 2.0 / l;
        xs = q[0]*s;
        ys = q[1]*s;
        zs = q[2]*s;
        
        wx = q[3]*xs;
        wy = q[3]*ys;
        wz = q[3]*zs;
        
        xx = q[0]*xs;
        xy = q[0]*ys;
        xz = q[0]*zs;
        
        yy = q[1]*ys;
        yz = q[1]*zs;
        zz = q[2]*zs;

        m[0*4+0] = 1.0 - (yy + zz);
        m[1*4+0] = xy - wz;
        m[2*4+0] = xz + wy;
        m[0*4+1] = xy + wz;
        m[1*4+1] = 1.0 - (xx + zz);
        m[2*4+1] = yz - wx;
        m[0*4+2] = xz - wy;
        m[1*4+2] = yz + wx;
        m[2*4+2] = 1.0 - (xx + yy);
        
        m[0*4+3] = 0.0;
        m[1*4+3] = 0.0;
        m[2*4+3] = 0.0;
        m[3*4+0] = m[3*4+1] = m[3*4+2] = 0.0;
        m[3*4+3] = 1.0;
        return m


