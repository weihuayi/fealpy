import numpy as np

from fealpy.decorator import cartesian

# 定义一个 PDE 的模型类
class PDEModel:
    def domain(self):
        """
        @brief 得到 PDE 模型的区域
        @return: 表示 PDE 模型的区域的列表
        """
        return np.array([0, 1, 0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = 3*np.pi**2*np.cos(np.pi*x)*np.cos(np.pi*y)*np.cos(np.pi*z)
        return val

    @cartesian
    def gradient(self, p):
        pi = np.pi
        sin = np.sin
        cos = np.cos
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = -pi*sin(pi*x)*cos(pi*y)*cos(pi*z)
        val[..., 1] = -pi*cos(pi*x)*sin(pi*y)*cos(pi*z)
        val[..., 2] = -pi*cos(pi*x)*cos(pi*y)*sin(pi*z)
        return val
        
    @cartesian    
    def dirichlet(self, p):
        return self.solution(p)

pde = PDEModel()
domain = pde.domain()

# 做一些测试
print('domain :', domain)
print(pde.solution(np.zeros(3)) + 1. < 1e-12)
print(pde.solution(np.ones(3)) + 1 < 1e-12)
print(pde.solution(np.array([0, 0, 1])) - 1 < 1e-12)

from fealpy.mesh import UniformMesh3d
import matplotlib.pyplot as plt 

nx = 5
ny = 5
nz = 5
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
hz = (domain[5] - domain[4])/nz
mesh = UniformMesh3d((0, nx, 0, ny, 0, nz), h=(hx, hy, hz), origin=(domain[0], domain[2], domain[4]))

# 画出网格
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1)
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True) 
mesh.find_cell(axes, showindex=True)
plt.title("Grid Image")
plt.show()
