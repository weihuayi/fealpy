import numpy as np

from fealpy.decorator import cartesian, barycentric

# 定义一个 PDE 的模型类
class PDEModel:
    @cartesian
    def domain(self):
        """
        @brief 得到 PDE 模型的区域
        @return: 表示 PDE 模型的区域的列表
        """
        return np.array([0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """
        @brief 计算 PDE 模型的精确解
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点的精确解
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi*x)*np.sin(pi*y)
        return val 
    
    @cartesian
    def source(self, p):
        """
        @brief: 计算 PDE 模型的原项 
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点处的源项
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.sin(pi*x)*np.sin(pi*y)
        return val
    
    @cartesian
    def gradient(self, p):
        """
        @brief: 计算 PDE 模型的真解的梯度
        @param p: 自标量 x,y 的数组
        @return: PDE 模型在给定点处真解的梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)
        return val
        
    @cartesian    
    def dirichlet(self, p):
        return self.solution(p)
   
pde = PDEModel()
domain = pde.domain()

# 做一些测试
print('domain :', domain)
print(pde.solution(np.zeros(2)) < 1e-12)
print(pde.solution(np.ones(2)) < 1e-12)
print(pde.solution(np.array([0, 1])) < 1e-12)

from fealpy.mesh import UniformMesh2d
import matplotlib.pyplot as plt

# 定义网格间距
hx = 0.2
hy = 0.2
# 根据区域和网格间距计算网格点的数量
nx = int((domain[1] - domain[0])/hx)
ny = int((domain[3] - domain[2])/hy)
mesh = UniformMesh2d((0, nx, 0, ny), h=(hx, hy), origin=(domain[0], domain[2]))
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

# 画出网格
fig = plt.figure(1)
axes = fig.gca()
mesh.add_plot(axes, aspect=1)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True) 
mesh.find_cell(axes, showindex=True)
plt.title("Grid Image")
plt.show()

# 将36个点处的离散解插值到网格节点上
uI = mesh.interpolate(pde.solution, 'node')

# 创建一个 figure，并设置当前坐标轴已进行绘图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(2)
axes = fig.add_subplot(111, projection='3d')
# show_function 函数在网格上绘制插值函数
mesh.show_function(axes, uI)
plt.title("Image of the function value of the right end term at the grid node")
plt.show()

# 画出真解的图像
x = np.linspace(0, 1, 101)
y = np.linspace(0, 1, 101)
X, Y = np.meshgrid(x, y)
p = np.array([X, Y]).T
Z = pde.solution(p)

fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='jet')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("Image of the ture solution")
plt.show()

A = mesh.laplace_operator()
print("未处理边界的矩阵A:\n", A.toarray())
uh = mesh.function()  # uh.shape = (6, 6)
f = mesh.interpolate(pde.source, 'node')
A, f = mesh.apply_dirichlet_bc(gD=pde.dirichlet, A=A, f=f)
print("处理完边界之后的矩阵A:\n", A.toarray())
print("处理完边界之后的右端项f:", f)

from scipy.sparse.linalg import spsolve

uh = mesh.function().reshape(-1, ) 
uh[:] = spsolve(A, f) # 返回网格节点处的数值解
print("数值解uh:", uh[:])
fig = plt.figure(4)
axes = fig.add_subplot(111, projection='3d')
mesh.show_function(axes, uh.reshape(6, 6))
plt.title("Numerical solution after processing the boundary conditions")
plt.show()


############################# 计算误差 ########################################
et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{l2}$']
eu = np.zeros(len(et), dtype=np.float64) 
uh = uh.reshape(6, 6)
eu[0], eu[1], eu[2] = mesh.error(pde.solution, uh)
et = np.array(et)
print(np.vstack((et, eu)))
print("----------------------------------------------------------------------")

# grad_uh = mesh.gradient(f=uh[:], order=1)
# print("grad_uh:\n", grad_uh.shape)
# print("----------------------------------------------------------------------")

# egradt = ['$|| \nabla u - \nabla u_h||_{\infty}$', '$|| \nabla u - \nabla u_h||_{0}$', '$|| \nabla u - \nabla u_h ||_{l2}$'] 
# egradu = np.zeros(len(et), dtype=np.float64) 
# egradu[0], egradu[1], egradu[2] = mesh.error(pde.gradient, grad_uh)
# egradt = np.array(egradt)
# print(np.vstack((egradt, egradu)))
# print("----------------------------------------------------------------------")
##############################  测试收敛阶 ###################################
maxit = 5
em = np.zeros((len(et), maxit), dtype=np.float64)
# egradm = np.zeros((len(et), maxit), dtype=np.float64) 

for i in range(maxit):
    A = mesh.laplace_operator() 
    uh = mesh.function().reshape(-1, ) 
    f = mesh.interpolate(pde.source, 'node')
    A, f = mesh.apply_dirichlet_bc(gD=pde.dirichlet, A=A, f=f)
    uh[:] = spsolve(A, f) 
  #  grad_uh = mesh.gradient(f=uh[:], order=1)
    NN = mesh.number_of_nodes()
    print("NN:", NN)
    uh = uh.reshape(int(np.sqrt(NN)), int(np.sqrt(NN)))
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)
   # egradm[0, i], egradm[1, i], egradm[2, i] = mesh.error(pde.gradient, grad_uh)

    if i < maxit:
        mesh.uniform_refine()

print("em:\n", em)
print("em_ratio:", em[:, 0:-1]/em[:, 1:])
# print("egradm:\n", egradm)
#print("egradm_ratio:", egradm[:, 0:-1]/egradm[:, 1:])

