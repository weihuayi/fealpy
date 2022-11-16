import numpy as np
import matplotlib.pyplot as plt

from TrussModel import Truss_3d

from fealpy.functionspace.Function import Function
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate

from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spsolve

class TrussSimulator():
    def __init__(self, pde, mesh):
        self.pde = pde
        self.mesh = mesh
        self.GD = mesh.geo_dimension()

    def edge_to_dof(self):
        mesh = self.mesh
        edge = mesh.entity('edge')
        GD = self.GD
        NN = mesh.number_of_nodes()

        edge2dof = np.zeros((edge.shape[0], 2*GD), dtype=np.int_)
        for i in range(GD):
            edge2dof[:, i::GD] = edge + NN*i
        return edge2dof

    def striff_matix(self):
        """

        Notes
        -----
        组装刚度矩阵

        """
        mesh = self.mesh
        GD = self.GD
        NN = mesh.number_of_nodes()
        NE= mesh.number_of_edges()

        E = self.pde.E
        A = self.pde.A
        l = self.mesh.edge_length().reshape(-1, 1)
        
        k = np.array([[1, -1], [-1, 1]], dtype=np.float64)
        R = np.zeros((NE, 2, GD*2), dtype=np.float64)
        tan = mesh.unit_edge_tangent()
        R[:, 0, :GD] = tan
        R[:, 1, -GD:] = tan
        K = np.einsum('ijk, jm, imn->ikn', R, k, R)
        K *= E*A
        K /= l[:, None]

        edge2dof = self.edge_to_dof()

        I = np.broadcast_to(edge2dof[:, :, None], shape=K.shape)
        J = np.broadcast_to(edge2dof[:, None, :], shape=K.shape)

        M = csr_matrix((K.flat, (I.flat, J.flat)), shape=(NN*GD, NN*GD))
        return M

    def source_vector(self):
        NN = self.mesh.number_of_nodes()
        GD = self.GD
        shape = (NN, GD)
        b = np.zeros(shape, dtype=np.float64)
        
        node = self.mesh.entity('node')
        isDof = self.pde.is_force_boundary(node)
        b[isDof] = self.pde.force()
        return b

    def function(self, dtype=np.float64):
        NN = self.mesh.number_of_nodes()
        dim = self.GD
        array = np.zeros((NN, dim), dtype=dtype)
        return Function(self, dim=dim, array=array, 
                coordtype='barycentric', dtype=dtype)

    def dirichlet_bc(self, M, F):
        NN = self.mesh.number_of_nodes()
        GD = self.GD
        shape = (NN, GD)
        uh = self.function()
        
        node = self.mesh.entity('node')
        isDDof = self.pde.is_dirichlet_boundary(node)
        isDDof = np.tile(isDDof, GD)
        F = F.T.flat
        x = uh.T.flat
        F -=M@x
        bdIdx = np.zeros(M.shape[0], dtype=np.int_)
        bdIdx[isDDof] = 1
        Tbd = spdiags(bdIdx, 0, M.shape[0], M.shape[0])
        T = spdiags(1-bdIdx, 0, M.shape[0], M.shape[0])
        M = T@M@T + Tbd
        F[isDDof] = x[isDDof]
        return M, F 

scale = 1
pde = Truss_3d()
mesh = pde.init_mesh()
simulator = TrussSimulator(pde, mesh)

uh = simulator.function()
M = simulator.striff_matix()
F = simulator.source_vector()
M, F = simulator.dirichlet_bc(M, F)
uh.T.flat[:] = spsolve(M, F)

print('uh:', uh)
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d') 
mesh.add_plot(axes)

mesh.node += scale*uh
mesh.add_plot(axes, nodecolor='b', edgecolor='m')
plt.show()
