import numpy as np
from scipy.sparse.linalg import spsolve
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags
from scipy.sparse import diags, lil_matrix
from scipy.sparse import vstack

Re = 1
nu = 1/Re

class taylor_greenData:
    def __init__(self, D=[0,2*np.pi,0,2*np.pi], T=[0, 5]):
        """
        @brief 模型初始化函数
        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        """
        self._domain = D 
        self._duration = T 
    
    def domain(self):
        """
        @brief 空间区间
        """
        return self._domain
    
    def duration(self):
        """
        @brief 时间区间
        """
        return self._duration 
    
    def solution_u(self, m, t):
        """
        @brief 真解函数u

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = np.cos(x)*np.sin(y)*Ft
        return val

    def init_solution_u(self, m):
        """
        @brief 初始解

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解函数值
        """
        x = m[..., 0]
        y = m[..., 1]
        val = np.cos(x)*np.sin(y)
        return val
    
    def solution_v(self, m, t):
        """
        @brief 真解函数u

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = -np.sin(x)*np.cos(y)*Ft
        return val

    def init_solution_v(self, m):
        """
        @brief 初始解

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解函数值
        """
        x = m[..., 0]
        y = m[..., 1]
        val = -np.sin(x)*np.cos(y)
        return val
    
    def solution_p(self, m, t):
        """
        @brief 真解函数p

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = -((np.cos(2*x)+np.cos(2*y))*Ft**2)/4
        return val
    
    def init_solution_p(self, m):
        """
        @brief 初始解

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解函数值
        """
        x = m[..., 0]
        y = m[..., 1]
        val = -(np.cos(2*x)+np.cos(2*y))/4
        return val
    
    def source_F(self,nu,t):
        """
        @brief 体积力
        @param[in] t float, 时间点 
        """
        return np.exp(-2*nu*t)

    def source(self, m, t):
        """
        @brief 方程右端项 

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 方程右端函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = np.zeros(m.shape, dtype=np.float64)
        val[..., 0] = -2*nu*np.cos(x)*np.sin(y)*Ft+np.sin(2*x)*Ft**2
        val[..., 1] = 2*nu*np.sin(x)*np.cos(y)*Ft+np.sin(2*y)*Ft**2
        return val
    
    def gradient_u(self, m, t):
        """
        @brief 真解u导数 

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解导函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = np.zeros(m.shape, dtype=np.float64)
        val[..., 0] = -np.sin(x)*np.sin(y)*Ft
        val[..., 1] = np.cos(x)*np.cos(y)*Ft
        return val
    
    def gradient_u(self, m, t):
        """
        @brief 真解u导数 

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解导函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = np.zeros(m.shape, dtype=np.float64)
        val[..., 0] =-np.cos(x)*np.cos(y)*Ft
        val[..., 1] = np.sin(x)*np.sin(y)*Ft
        return val

    def gradient_p(self, m, t):
        """
        @brief 真解p导数 

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解导函数值
        """
        Ft = np.exp(-2*nu*t)
        x = m[..., 0]
        y = m[..., 1]
        val = np.zeros(m.shape, dtype=np.float64)
        val[..., 0] = (np.sin(2*x)*Ft**2)/2
        val[..., 1] = (np.sin(2*y)*Ft**2)/2
        return val
    
    def dirichlet_u(self, m, t):
        """
        @brief Dirichlet 边界条件

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 
        """
        return self.solution_u(m, t)
    
    def dirichlet_v(self, m, t):
        """
        @brief Dirichlet 边界条件

        @param[in] m numpy.ndarray, 空间点
        @param[in] t float, 时间点 
        """
        return self.solution_v(m, t)

#PDE 模型 
pde = taylor_greenData()
domain = pde.domain()

#空间离散
nx = 4
ny = 4
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh_u = UniformMesh2d([0, nx, 0, ny-1], h=(hx, hy), origin=(domain[0], domain[2]+hy/2))
mesh_v = UniformMesh2d([0, nx-1, 0, ny], h=(hx, hy), origin=(domain[0]+hx/2, domain[2]))
mesh_p = UniformMesh2d([0, nx-1, 0, ny-1], h=(hx, hy), origin=(domain[0]+hx/2, domain[2]+hy/2))

#时间离散
duration = pde.duration()
nt = 128
tau = (duration[1] - duration[0])/nt

'''
fig = plt.figure()
axes = fig.gca()
mesh_u.add_plot(axes)
#mesh_v.add_plot(axes)
#mesh_p.add_plot(axes)
mesh_u.find_node(axes, color='r')
#mesh_v.find_node(axes, color='b')
#mesh_p.find_node(axes, color='g')
#mesh_u.find_node(axes, showindex=True, fontsize=12, color='r')
#mesh_v.find_node(axes, showindex=True, fontsize=12, color='b')
#mesh_p.find_node(axes, showindex=True, fontsize=12, color='g')
#mesh_u.find_edge(axes, showindex=True, fontsize=12, fontcolor='g') 
#mesh_u.find_cell(axes, showindex=True, fontsize=12, fontcolor='r')
plt.show()
'''
'''
# 可视化模型
x = np.linspace(domain[0], domain[1], nx)
y = np.linspace(domain[2], domain[3], ny)
X, Y = np.meshgrid(x, y)
# 初始化时间
t = 0.0
# 计算解值
U = pde.solution_u(np.dstack((X, Y)), t)
U = U.reshape((ny, nx))  
V = pde.solution_v(np.dstack((X, Y)), t)
V = V.reshape((ny, nx))
P = pde.solution_p(np.dstack((X, Y)), t)
P = P.reshape((ny, nx))
# 创建一个图
plt.figure(figsize=(12, 6))
# 画 solution_u 的三维曲面图
ax = plt.subplot(131, projection='3d')
ax.plot_surface(X, Y, U, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u')
ax.set_title('u')
# 画 solution_v 的三维曲面图
ax = plt.subplot(132, projection='3d')
ax.plot_surface(X, Y, V, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('v')
ax.set_title('v')
# 画 solution_p 的三维曲面图
ax = plt.subplot(133, projection='3d')
ax.plot_surface(X, Y, P, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('p')
ax.set_title('p')

plt.tight_layout()
plt.show()

# 画真解
x = np.linspace(0, 2 * np.pi, 100)
y = np.linspace(0, 2 * np.pi, 100)
t = 0
X, Y = np.meshgrid(x, y)

# 计算函数值
Re = 1
nu = 1 / Re
u = np.cos(X) * np.sin(Y) * np.exp(-2 * t * nu)
v = -np.sin(X) * np.cos(Y) * np.exp(-2 * t * nu)
p = -(np.cos(2 * X) + np.cos(2 * Y)) * (np.exp(-2 * t * nu))**2 / 4
# 创建三维图形
fig = plt.figure(figsize=(12, 6))
# 绘制 u 函数
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, u, cmap='viridis')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('u(x, y, t)')
# 绘制 v 函数
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, v, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('v(x, y, t)')
# 绘制 p 函数
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, p, cmap='viridis')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('p(x, y, t)')

plt.show()
'''
#取网格u,v,p的节点位置
nodes_u = mesh_u.entity('node') #[20.2]
nodes_v = mesh_v.entity('node') #[20,2]
nodes_p = mesh_p.entity('node') #[16,2]

#计算网格u,v,p节点的总数
num_nodes_u = nodes_u.shape[0] #20
num_nodes_v = nodes_v.shape[0] #20
num_nodes_p = nodes_p.shape[0] #16

def solution_u_0(p):
    return pde.solution_u(p,t=0)
def solution_v_0(p):
    return pde.solution_v(p,t=0)
def solution_p_0(p):
    return pde.solution_p(p,t=0)

solution_u = mesh_u.interpolate(solution_u_0) #[4,5]
solution_u_values = solution_u.reshape(-1)
solution_v = mesh_v.interpolate(solution_v_0)
solution_v_values = solution_v.reshape(-1)
solution_p = mesh_p.interpolate(solution_p_0)
solution_p_values = solution_p.reshape(-1)

def grad_ux(mesh):
    Nrow = mesh.node.shape[1]
    Ncol = mesh.node.shape[0]
    N = Nrow*Ncol

    result = diags([1, -1],[Nrow, -Nrow],(N,N), format='csr')

    return result
np.set_printoptions(linewidth=1000)
#print(grad_ux(mesh_u).toarray())

def grad_uy(mesh):
    Nrow = mesh.node.shape[1]
    Ncol = mesh.node.shape[0]
    N = Nrow*Ncol
    result = diags([1, -1],[1, -1],(N,N), format='lil')

    index = np.arange(0, N, Nrow)
    result[index, index] = 2
    result[index, index+1] = 2/3
    result[index[1:], index[1:]-1] = 0

    index = np.arange(Nrow-1, N, Nrow)
    result[index, index] = -2
    result[index, index-1] = -2/3
    result[index[:-1], index[:-1]+1] = 0

    return result
np.set_printoptions(linewidth=1000)
#print(grad_uy(mesh_u).toarray())
    
def Tuv(mesh):
    Nrow = mesh.node.shape[1]
    Ncol = mesh.node.shape[0]
    N = Nrow*Ncol
    result = np.zeros((N,N))
    index = np.arange(N-Nrow)
    result[index,index+index//Nrow] = 1
    result[index,index+index//Nrow+1] = 1
    result[index,index+index//Nrow-Nrow-1] = 1
    result[index,index+index//Nrow-Nrow] = 1
    return result
np.set_printoptions(linewidth=1000)
#print(Tuv(mesh_u))

def laplace_u(mesh):
    Nrow = mesh.node.shape[1]
    Ncol = mesh.node.shape[0]
    N = Nrow*Ncol
    result = diags([-4, 1, 1, 1, 1],[0, 1, -1, Nrow, -Nrow],(N,N), format='lil')

    index = np.arange(0,N, Nrow)
    result[index, index] = -6
    result[index[2:]-1, index[2:]-1] = -6
    result[index[1:], index[1:]-1] = 0
    result[index[2:]-1, index[2:]] = 0
    result[index, index+1] = 4/3
    result[index[2:]-1, index[2:]-2] = 4/3

    return result
np.set_printoptions(linewidth=1000)
#print(laplace_u(mesh_u).toarray())

def grand_uxp(mesh):
    Nrow = mesh.node.shape[1]
    Ncol = mesh.node.shape[0]
    N = Nrow*Ncol
    result = diags([1, -1],[0, -Nrow],(N,N), format='lil')
    A = lil_matrix((Nrow, N))
    result1 = vstack([result, A], format='lil')
    
    return result1
np.set_printoptions(linewidth=1000)
#print(grand_uxp(mesh_p).toarray())

def source_F(pde,mesh_p,t):
    nodes_p = mesh_p.entity('node')
    num_nodes_p = nodes_p.shape[0]
    # 初始化数组，存储体积力
    result = np.zeros((num_nodes_p, 1))
    source_p = pde.source_F(nu, t) 
    result += source_p * np.ones((num_nodes_p, 1))

    return result
#print(source_F(pde,mesh_p,t=1))

grandux = grad_ux(mesh_u)/(2*hx) @ solution_u_values
granduy = grad_uy(mesh_u)/(2*hy) @ solution_u_values
grandvx = Tuv(mesh_u) @ solution_v_values
AD_xu = solution_u_values * grandux + grandvx * granduy
#print(AD_xu)


