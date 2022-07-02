import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.mesh import IntervalMesh
from fealpy.quadrature import IntervalQuadrature
from fealpy.decorator import cartesian

class SinData:
    def __init__(self):
        pass

    @cartesian
    def solution(self, p):
        """ the exact solution

        Parameters
        ---------
        p : numpy.ndarray
            (..., 1)
        """
        x = p[..., 0]
        val = np.sin(np.pi*x)
        return val 

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution

        Parameters
        ---------
        p : numpy.ndarray
            (..., 1)
        """
        x = p[..., 0]
        pi = np.pi
        val = pi*np.cos(pi*x)
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0] # (NQ, NC)
        val = np.pi**2*np.sin(np.pi*x)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

pde = SinData()

node = np.array([
    [0.0],
    [1.0]], dtype=np.float64)

cell = np.array([
    [0, 1]], dtype=np.int_)

mesh = IntervalMesh(node, cell)

mesh.uniform_refine(n=2)

node = mesh.entity('node')
cell = mesh.entity('cell')


l = mesh.entity_measure('cell')
gphi = mesh.grad_lambda() # (NC, 2, 1)

S = np.einsum('cim, cjm, c->cij', gphi, gphi, l)  # (NC, 2, 2)

# cell.shape = (NC, 2) --> (NC, 2, 1) --> (NC, 2, 2)
I = np.broadcast_to(cell[:, :, None], shape=S.shape)
# cell.shape = (NC, 2) --> (NC, 1, 2) --> (NC, 2, 2)
J = np.broadcast_to(cell[:, None, :], shape=S.shape)

NN = mesh.number_of_nodes()
S = csr_matrix((S.flat, (I.flat, J.flat)), shape=(NN, NN))

qf = IntervalQuadrature(3)
# bcs.shape = (NQ, 2)
# ws.shape = (NQ, )
bcs, ws = qf.get_quadrature_points_and_weights()
phi = bcs


#M = np.einsum('q, qi, qj, c->cij', ws, phi, phi, l)
#M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(NN, NN))
#A = S+M

# (f, lambda_0) (f, lambda_1)

# bcs.shape == (NQ, 2)
# node.shape == (NN, 1)
# cell.shape == (NC, 2)
# node[cell].shape == (NC, 2, 1)
# ps.shape == (NQ, NC, 1)
ps = np.einsum('qi, cim->qcm', bcs, node[cell]) 
val = pde.source(ps) # (NQ, NC)
# bb.shape ==(NC, 2)
bb = np.einsum('q, qc, qi, c->ci', ws, val, bcs, l)

F = np.zeros(NN, dtype=np.float64)
np.add.at(F, cell, bb)


# S u = F
# isBdNode.shape == (NN, )
isBdNode = mesh.ds.boundary_node_flag()
isInteriorNode = ~isBdNode

uh = np.zeros(NN, dtype=np.float64)

uh[isInteriorNode] = spsolve(S[:, isInteriorNode][isInteriorNode, :], F[isInteriorNode])

uI = pde.solution(node)

e = np.max(np.abs(uI - uh))
print(e)





fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, markersize=50)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()

