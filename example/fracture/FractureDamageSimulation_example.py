import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.TriMesher import distmesh2d
from fealpy.geometry import dcircle, drectangle
from fealpy.geometry import ddiff, huniform

from fealpy.functionspace import LagrangeFiniteElementSpace

class Brittle_Facture_model():
    def __init__(self):
        self.E = 200 # 杨氏模量
        self.nu = 0.2 # 泊松比
        self.Gc = 1 # 材料的临界能量释放率
        self.l0 = 0.02 # 尺度参数，断裂裂纹的宽度

        self.mu = self.E / (1 + self.nu) / 2.0 # 剪切模量
        self.la = self.E * self.mu / (1 + self.nu) / (1- 2*self.nu) # 拉梅常数
        self.kappa = self.la + 2 * self.mu /3 # 压缩模量

        self.eps = 1e-6 # 极小值，用来保持数值稳定性


    def init_mesh(self):
        """
        @brief 生成实始网格, 正方形区域，内部有一个半径为 0.2 的洞
        """
        fd1 = lambda p: dcircle(p, [0.5, 0.5], 0.2)
        fd2 = lambda p: drectangle(p, [0.0, 1, 0.0, 1])
        fd = lambda p: ddiff(fd2(p), fd1(p))

        fh = huniform
        bbox = [0, 1, 0, 1]
        pfix = np.array([
            (0.0, 0.0), 
            (1.0, 0.0), 
            (1.0, 1.0),
            (0.0, 1)],dtype=np.float64)

        mesh = distmesh2d(hmin=0.02, fd=fd, fh=fh, bbox=bbox, pfix=pfix)
        return mesh

    def top_boundary_disp(self):
        """
        @brief Dirichlet 边界条件，位移边界条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return np.concatenate((np.linspace(0, 70e-3, 6)[1:], np.linspace(70e-3,
            125e-3, 26)[1:]))

    def top_disp_direction(self):
        """
        @brief 上边界位移的方向
        Notes
        -----
        位移方向沿 (0, 1) 方向，即仅在 y 方向的位移变化
        """
        return np.array([0, 1], np.float_)

    def is_top_boundary(self, p):
        """
        @brief 标记上边界, y = 1 时的边界点
        """
        return np.abs(p[..., 1] - 1) < 1e-12 

    def is_inter_boundary(self, p):
        """
        @brief 标记内部边界, 内部圆的点
        Notes
        -----
        内部圆周的点为 DirichletBC，相场值和位移均为 0
        """
        return np.abs((p[..., 0]-0.5)**2 + np.abs(p[..., 1]-0.5)**2 - 0.04) < 0.001


model = Brittle_Facture_model()
mesh = model.init_mesh()
node = mesh.entity('node')
for i in range(node.shape[0]):
    if model.is_inter_boundary(node)[i] == True:
        print(node[i])

fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes)
plt.show()
