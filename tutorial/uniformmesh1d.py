import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.mesh import UniformMesh1d

# 定义一个 PDE 的模型类
class PDEModel:
    def domain(self):
        return [0, 1]

    def solution(self, x):
        return np.sin(4*np.pi*x)

    def gradient(self, x):
        return 4*np.pi*np.cos(4*np.pi*x)
        
    def source(self, x):
        return 16*np.pi**2*np.sin(4*np.pi*x)

pde = PDEModel()
domain = pde.domain()

nx = 10
maxit = 4
et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{1}$']
em = np.zeros((len(et), maxit), dtype=np.float64)
for i in range(maxit):
    hx = (domain[1] - domain[0])/nx
    mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])
    A = mesh.laplace_operator_with_dbc()
    uI = mesh.interpolation(pde.solution, 'node')
    F = mesh.interpolation(pde.source, 'node')
    F[0] = uI[0]
    F[-1] = uI[-1]
    uh = spsolve(A, F)
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)
    nx *= 2

print(em)
    
fig = plt.figure()
axes = fig.gca()
mesh.show_function(axes, uh)
plt.show()
