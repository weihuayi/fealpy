import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace.Function import Function
from fealpy.mesh.TrussMesh import TrussMesh

from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spsolve

A = 2000 # 横截面积 mm^2
E = 1500 # 弹性模量 ton/mm^2

# 构造网格
d1 = 952.5 # 单位 mm
d2 = 2540
h1 = 5080
h2 = 2540
node = np.array([
    [-d1, 0, h1], [d1, 0, h1], [-d1, d1, h2], [d1, d1, h2],
    [d1, -d1, h2], [-d1, -d1, h2], [-d2, d2, 0], [d2, d2, 0],
    [d2, -d2, 0], [-d2, -d2, 0]], dtype=np.float64)
edge = np.array([
    [0, 1], [3, 0], [1, 2], [1, 5], [0, 4], 
    [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
    [4, 3], [2, 3], [4, 5], [2, 9], [6, 5], 
    [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
    [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=np.int_)
mesh = TrussMesh(node, edge)

GD = mesh.geo_dimension()
NN = mesh.number_of_nodes()
NE= mesh.number_of_edges()

def striff_matix(A, E):
    """

    Notes
    -----
    组装刚度矩阵

    """
    l = mesh.edge_length().reshape(-1, 1)
    
    tan = mesh.unit_edge_tangent()
    R = np.einsum('ik, im->ikm', tan, tan)
    K = np.zeros((NE, GD*2, GD*2), dtype=np.float64)
    K[:, :GD, :GD] = R
    K[:, -GD:, :GD] = -R
    K[:, :GD, -GD:] = -R
    K[:, -GD:, -GD:] = R
    K *= E*A
    K /= l[:, None]

    edge = mesh.entity('edge')

    edge2dof = np.zeros((edge.shape[0], 2*GD), dtype=np.int_)
    for i in range(GD):
        edge2dof[:, i::GD] = edge + NN*i

    I = np.broadcast_to(edge2dof[:, :, None], shape=K.shape)
    J = np.broadcast_to(edge2dof[:, None, :], shape=K.shape)

    M = csr_matrix((K.flat, (I.flat, J.flat)), shape=(NN*GD, NN*GD))
    return M

def source_vector(isDof, force):
    shape = (NN, GD)
    b = np.zeros(shape, dtype=np.float64)
    
    b[isDof] = force
    return b

def dirichlet_bc(M, F, isDDof):
    shape = (NN, GD)
    uh = np.zeros(shape, dtype=np.float64)
    
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

uh = np.zeros((NN, GD), dtype=np.float64)
M = striff_matix(A, E)
F = source_vector(np.abs(node[..., 2]) == 5080, force=np.array([0,
    900, 0]))

M, F = dirichlet_bc(M, F, np.abs(node[..., 2]) < 1e-12)
uh.T.flat = spsolve(M, F)

fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d') 
mesh.add_plot(axes)

mesh.node += uh
mesh.add_plot(axes, nodecolor='b', edgecolor='m')
plt.show()


