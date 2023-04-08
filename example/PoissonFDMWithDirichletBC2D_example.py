import numpy as np

from fealpy.pde.elliptic_2d import SinSinPDEData

# PDE 模型
pde = SinSinPDEData()
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
print("NN:", mesh.number_of_nodes())
print("NE:", mesh.number_of_edges())
print("NF:", mesh.number_of_faces())
print("NC:", mesh.number_of_cells())
print("gd:", mesh.geo_dimension())
print("td:", mesh.top_dimension())
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

# 画出网格
fig = plt.figure(1)
axes = fig.gca()
mesh.add_plot(axes)
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

uh = mesh.function() 
uh.flat[:] = spsolve(A, f) # 返回网格节点处的数值解
print("数值解uh:", uh[:])
fig = plt.figure(4)
axes = fig.add_subplot(111, projection='3d')
mesh.show_function(axes, uh)
plt.title("Numerical solution after processing the boundary conditions")
plt.show()


############################# 计算误差 ########################################
et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{l2}$']
eu = np.zeros(len(et), dtype=np.float64) 
eu[0], eu[1], eu[2] = mesh.error(pde.solution, uh)
et = np.array(et)
print(np.vstack((et, eu)))
print("----------------------------------------------------------------------")

##############################  测试收敛阶 ###################################
maxit = 5
em = np.zeros((len(et), maxit), dtype=np.float64)

for i in range(maxit):
    A = mesh.laplace_operator() 
    uh = mesh.function()
    f = mesh.interpolate(pde.source, 'node')
    A, f = mesh.apply_dirichlet_bc(gD=pde.dirichlet, A=A, f=f)
    uh.flat[:] = spsolve(A, f) 
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)

    if i < maxit:
        mesh.uniform_refine()

print("em:\n", em)
print("em_ratio:", em[:, 0:-1]/em[:, 1:])

