import numpy as np 
import matplotlib.pyplot as plt 

from fealpy.decorator import cartesian, barycentric 
from fealpy.mesh import UniformMesh1d

class PDEModel:
    def domain(self):
        """
        @brief: Get the domain of the PDE model
        @return: A list representing the domain of the PDE model
        """
        return [0, 1]

    @cartesian    
    def solution(self, p):
        """
        @brief: Calculate the exact solution of the PDE model
        @param p: An array of the independent variable x
        @return: The exact solution of the PDE model at the given points
        """
        return np.sin(4*np.pi*p)
    
    @cartesian    
    def source(self, p):
        """
        @brief: Calculate the source term of the PDE model
        @param p: An array of the independent variable x
        @return: The source term of the PDE model at the given points
        """
        return 16*np.pi**2*np.sin(4*np.pi*p)
    
    @cartesian    
    def gradient(self, p):
        """
        @brief: Calculate the gradient of the exact solution of the PDE model
        @param p: An array of the independent variable x
        @return: The gradient of the exact solution of the PDE model at the given points
        """
        return 4*np.pi*np.cos(4*np.pi*p)

    @cartesian    
    def dirichlet(self, p):
        """
        @brief: 模型的 Dirichlet 边界条件
        """
        return self.solution(p)

pde = PDEModel()
domain = pde.domain()

# test
print('domain :', domain)
print(np.abs(pde.solution(0)) < 1e-12)
print(np.abs(pde.solution(1)) < 1e-12)
print(np.abs(pde.solution(1/8) - 1) < 1e-12)

hx = 0.1
nx = int((domain[1] - domain[0])/hx)
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

fig = plt.figure(1)
axes = fig.gca() 
mesh.add_plot(axes) 
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True) 
plt.title("Grid Image")
plt.show()

uI = mesh.interpolate(pde.solution, 'node') # 返回右端项在网格节点处的函数值

fig = plt.figure(2)
axes = fig.gca()
mesh.show_function(axes, uI)
plt.title("Image of the function value of the right end term at the grid node")
plt.show()

# 画出真解的图像
x = np.linspace(domain[0], domain[1], 100)
u = pde.solution(x)

fig = plt.figure(3)
axes = fig.gca()
axes.plot(x, u)
plt.title("Image of the ture solution")
plt.show()

A = mesh.laplace_operator()
print("未处理边界的矩阵A:\n", A.toarray())
uh = mesh.function() # 返回网格节点处的函数值，默认值为 0
f = mesh.interpolate(pde.source, 'node')

A, f = mesh.apply_dirichlet_bc(gD=pde.dirichlet, A=A, f=f)
print("处理完边界之后的矩阵A:\n", A.toarray())
print("处理完边界之后的右端项f:", f)

from scipy.sparse.linalg import spsolve

uh[:] = spsolve(A, f) # 返回网格节点处的数值解
fig = plt.figure(4)
axes = fig.gca()
mesh.show_function(axes, uh)
plt.title("Numerical solution after processing the boundary conditions")
plt.show()

############################# 计算误差 ########################################
et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{1}$']
eu = np.zeros(len(et), dtype=np.float64) 
eu[0], eu[1], eu[2] = mesh.error(pde.solution, uh)
et = np.array(et)
print(np.vstack((et, eu)))
print("----------------------------------------------------------------------")

grad_uh = mesh.gradient(f=uh[:], order=1)
print("grad_uh:\n", grad_uh)
print("----------------------------------------------------------------------")

egradt = ['$|| \nabla u - \nabla u_h||_{\infty}$', '$|| \nabla u - \nabla u_h||_{0}$', '$|| \nabla u - \nabla u_h ||_{1}$'] 
egradu = np.zeros(len(et), dtype=np.float64) 
egradu[0], egradu[1], egradu[2] = mesh.error(pde.gradient, grad_uh)
egradt = np.array(egradt)
print(np.vstack((egradt, egradu)))
print("----------------------------------------------------------------------")

##############################  测试收敛阶 ###################################
maxit = 5
em = np.zeros((len(et), maxit), dtype=np.float64)
egradm = np.zeros((len(et), maxit), dtype=np.float64) 

for i in range(maxit):
    A = mesh.laplace_operator() 
    uh = mesh.function() 
    f = mesh.interpolate(pde.source, 'node')
    A, f = mesh.apply_dirichlet_bc(gD=pde.dirichlet, A=A, f=f)
    uh[:] = spsolve(A, f) 
    grad_uh = mesh.gradient(f=uh[:], order=1)

    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)
    egradm[0, i], egradm[1, i], egradm[2, i] = mesh.error(pde.gradient, grad_uh)

    if i < maxit:
        mesh.uniform_refine()

print("em:\n", em)
print("em_ratio:", em[:, 0:-1]/em[:, 1:])
print("egradm:\n", egradm)
print("egradm_ratio:", egradm[:, 0:-1]/egradm[:, 1:])

