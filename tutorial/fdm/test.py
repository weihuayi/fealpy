import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.IntervalMesh import IntervalMesh
from fealpy.mesh import StructureIntervalMesh
from scipy.sparse.linalg import spsolve
from fealpy.pde.poisson_2d import CosCosData
from fealpy.pde.poisson_2d import SinSinData



class SinData:
    def __init__(self):
        pass

    def domain(self):
        return [0, 1]

    def init_mesh(self, n=1, meshtype='struct', h=0.1):
        d = self.domain()
        if meshtype is 'struct':
            mesh = StructureIntervalMesh(d, h)
            return mesh
        else:
            node = np.array(d, dtype=np.float)
            cell = np.array([(0, 1)], dtype=np.int)
            mesh = IntervalMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh

    def solution(self, p):
        """ the exact solution
        """
        u = np.sin(4*np.pi*p)
        return u

    def gradient(self, p):
        """ The gradient of the exact solution
        """
        pi = np.pi
        cos = np.cos

        val = 4*pi*cos(4*pi*p)
        return val[..., np.newaxis]

    def source(self, p):
        val = 16*np.pi**2*np.sin(4*np.pi*p)
        return val

    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)


maxit = 4
h = 0.1
# pde = SinData()
nx = 10 
ny = 10
pde = SinSinData()

for i in range(maxit):
    mesh = pde.init_mesh(nx=nx, ny=ny, meshtype='stri')

    NN = mesh.number_of_nodes()
    node = mesh.entity('node')

    u = np.zeros(NN, dtype=np.float)
    b = pde.source(node)

    A = mesh.laplace_operator()

    isBdNode = mesh.ds.boundary_node_flag()

    u[~isBdNode] = spsolve(A[~isBdNode, :][:, ~isBdNode], b[~isBdNode])

    e = np.max(np.abs(u - pde.solution(node)))

    print(e)

    if i < maxit-1:
        # h /= 2
        nx *= 2
        ny *= 2

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, markersize=12)
plt.show()
