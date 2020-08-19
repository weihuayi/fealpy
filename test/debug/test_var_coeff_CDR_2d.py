import sys
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, show_error_table

class CDRMODEL:
    """
	Equation:
    -\\nu\\nabla\cdot(A(x)\\nabla u + b(x)u) + c(x)u = f in \Omega
	
	B.C.:
	u = g_D on \partial\Omega
	
	Exact Solution:
        u = sin(pi*x)*sin(pi*y)+x^5+y^5+1
	
	Coefficients:
    \\nu = 1, 1e-3, 1e-9
	A(x) = [1+x^2, x*y;
			x*y, 1+y^2]
	b(x) = [-cos(2*pi*x); -sin(2*pi*y)] or [-1; -1]
	c(x) = exp(x+y)
    """
    def __init__(self):
        pass

    def domain(self):
        return np.array([0, 1, 0, 1])
    
    @cartesian
    def solution(self, p):
        """ 
		The exact solution 
        Parameters
        ---------
        p : 


        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi*x) * np.sin(pi*y) + x**5 + y**5 + 1
        return val # val.shape == x.shape

    @cartesian
    def source(self, p):
        """ 
		The right hand side of convection-diffusion-reaction equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(2*pi*x) * (5*x**4 + pi*np.cos(pi*x)*np.sin(pi*y))  
        val -= 3 * y * (5*y**4 + pi*np.cos(pi*y)*np.sin(pi*x))
        val -= 3 * x *(5*x**4 + pi*np.cos(pi*x)*np.sin(pi*y)) 
        val += np.sin(2*pi*y) * (5*y**4 + pi*np.cos(pi*y)*np.sin(pi*x)) 
        val -= (x**2+1) * (20*x**3 - pi*pi*np.sin(pi*x)*np.sin(pi*y))
        val -= (y**2+1) * (20*y**3 - pi*pi*np.sin(pi*x)*np.sin(pi*y))
        val += np.exp(x+y) * (np.sin(pi*x)*np.sin(pi*y) + x**5 + y**5 + 1)
        val += 2*pi*np.cos(2*pi*y) * (np.sin(pi*x)*np.sin(pi*y) + x**5 + y**5 + 1)
        val -= 2*pi*np.sin(2*pi*x) * (np.sin(pi*x)*np.sin(pi*y) + x**5 + y**5 + 1)
        val -= 2*x*y*pi*pi*np.cos(pi*x)*np.cos(pi*y)
        return val

    @cartesian
    def gradient(self, p):
        """ 
		The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y) + 5*x**4
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y) + 5*y**4
        return val # val.shape == p.shape

    @cartesian
    def diffusion_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        eips = 1
        shape = p.shape + (2, )
        val = np.zeros(shape, dtype=np.float64) 
        val[..., 0, 0] = eips*(1+x**2)
        val[..., 0, 1] = eips*x*y
        val[..., 1, 0] = val[..., 0, 1]
        val[..., 1, 1] = eips*(1+y**2)
        return val 

    @cartesian
    def convection_coefficient_ndf(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -np.cos(2*pi*x)
        val[..., 1] = -np.sin(2*pi*y)
        return val
    
    @cartesian
    def convection_coefficient_df(self, p):
        return np.array([-1.0, -1.0], dtype=np.float64)
    
    @cartesian
    def reaction_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.exp(x+y)
        return val

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)
		

p = 1
n = 4
maxit = 4

pde = CDRMODEL()
domain = pde.domain()
mf = MeshFactory()
mesh = mf.boxmesh2d(domain, nx=n, ny=n, meshtype='tri')
NDof = np.zeros(maxit, dtype=mesh.itype)

errorMatrix = np.zeros((2, maxit), dtype=mesh.ftype)
errorType = ['$|| u  - u_h ||_0$', '$|| \\nabla u - \\nabla u_h||_0$']

for i in range(maxit):
    print('Step:', i)
    space = LagrangeFiniteElementSpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    uh = space.function()
    A = space.stiff_matrix(c=pde.diffusion_coefficient)
    B = space.convection_matrix(c=pde.convection_coefficient_ndf)
    M = space.mass_matrix(c=pde.reaction_coefficient)
    F = space.source_vector(pde.source)
    A += B 
    A += M
    
    bc = DirichletBC(space, pde.dirichlet)
    A, F = bc.apply(A, F, uh)

    uh[:] = spsolve(A, F)

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh.value, power=2)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value,
            power=2)

    if i < maxit-1:
        mesh.uniform_refine()
		
# 函数解图像	
uh.add_plot(plt, cmap='rainbow')

# 收敛阶图像
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=10)

# 输出误差的 latex 表格
show_error_table(NDof, errorType, errorMatrix)
plt.show()
