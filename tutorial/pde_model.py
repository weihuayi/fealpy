
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D 画图必备


from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.quadrature import FEMeshIntegralAlg
from fealpy.functionspace import LagrangeFiniteElementSpace

class CosCosData:
    """
    -\Delta u = f
    u = cos(pi*x)*cos(pi*y)
    """
    def __init__(self):
        """
        构造函数
        """
        self.a = 10

    def domain(self):
        return np.array([0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """ The exact solution 
        Parameters
        ---------
        p : ndarray, (..., 2)--> (...)


        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        (2, )

        p = np.array([[0, 1], [0.5, 0.5], 
            [0.2, 0.3]], dtype=np.float64)
        (3, 2)
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape == y.shape

    @cartesian
    def source(self, p):
        """ The right hand side of Possion equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.cos(pi*x)*np.cos(pi*y)
        return val

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def flux(self, p):
        return -self.gradient(p)

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

    @cartesian
    def neumann(self, p, n):
        """ 
        Neuman  boundary condition

        Parameters
        ----------

        p: (NQ, NE, 2)
        n: (NE, 2)

        grad*n : (NQ, NE, 2)
        """
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        shape = len(val.shape)*(1, )
        kappa = np.array([1.0], dtype=np.float64).reshape(shape)
        val += self.solution(p) 
        return val, kappa

p = 1
q = 3
pde = CosCosData()
print(pde.a)

domain = pde.domain()
mesh = MF.boxmesh2d(domain, nx=10, ny=10, meshtype='tri')
intalg = FEMeshIntegralAlg(mesh, q)

c = intalg.mesh_integral(pde.solution, q=q, power=2)

print(c)



space = LagrangeFiniteElementSpace(mesh, p=p)
node = mesh.entity('node') # (NN, 2)
uI = pde.solution(node) # ndarray (NN, )

uI = space.function(array=uI) # 返回一个有限元函数对象

fig0 = plt.figure()
axes = fig0.gca()
mesh.add_plot(axes)

fig1 = plt.figure()
axes = fig1.gca(projection='3d')
uI.add_plot(axes, cmap='rainbow')
plt.show()


if False:
    box = [0, 1, 0, 1]
    mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype='tri')
    space = LagrangeFiniteElementSpace(mesh, p=p)

# 插值
    uI = space.interpolation(pde.solution) # 是个有限元函数，同时也是一个数组

    print('uI[0:10]:', uI[0:10]) # 打印前面 10 个自由度的值

    bc = np.array([1/3, 1/3, 1/3], dtype=mesh.ftype) # (3， )
    val0 = uI(bc) # (NC， )
    val1 = uI.grad_value(bc) # (NC, 2)

    print('val0[0:10]:', val0[1:10]) # 打 uI 在前面 10 个单元重心处的函数值
    print('val1[0:10]:', val1[1:10]) # 打 uI 在前面 10 个单元重心处的梯度值

# 插值误差

    error0 = space.integralalg.L2_error(pde.solution, uI)
    error1 = space.integralalg.L2_error(pde.gradient, uI.grad_value)
    print('L2:', error0, 'H1:', error1)

    fig = plt.figure()
    axes = fig.gca(projection='3d')
    uI.add_plot(axes, cmap='rainbow')
    plt.show()
