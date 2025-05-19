from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh
from fealpy.model import PDEDataManager
from fealpy.sparse import CSRTensor 
from scipy import sparse

pde = PDEDataManager('poisson').get_example('coscos')
domain = pde.domain()  # [-1, 1, -1, 1]
extent = [0, 2, 0, 2]
mesh = UniformMesh(domain, extent)

def bd_1(node):
    x = node[:, 0]
    y = node[:, 1]
    return x == -1

def dirichlet_bc(threshold, mesh):
    total_bd_idx = mesh.boundary_node_index()
    bd_node = mesh.node[total_bd_idx]
    bd_idx = threshold(bd_node)
    return bd_idx

x = dirichlet_bc(bd_1, mesh)

A_before_scipy = sparse.csr_matrix(bm.array(
    [[ 4., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
     [-1.,  4., -1.,  0., -1.,  0.,  0.,  0.,  0.],
     [ 0., -1.,  4.,  0.,  0., -1.,  0.,  0.,  0.],
     [-1.,  0.,  0.,  4., -1.,  0., -1.,  0.,  0.],
     [ 0., -1.,  0., -1.,  4., -1.,  0., -1.,  0.],
     [ 0.,  0., -1.,  0., -1.,  4.,  0.,  0., -1.],
     [ 0.,  0.,  0., -1.,  0.,  0.,  4., -1.,  0.],
     [ 0.,  0.,  0.,  0., -1.,  0., -1.,  4., -1.],
     [ 0.,  0.,  0.,  0.,  0., -1.,  0., -1.,  4.]]
))
A_before = CSRTensor(A_before_scipy.indptr, A_before_scipy.indices, A_before_scipy.data, A_before_scipy.shape)

f_before = bm.array([ 19.7392088, -19.7392088,  19.7392088, -19.7392088,  19.7392088, -19.7392088,
                      19.7392088, -19.7392088,  19.7392088])

A_after_none_scipy = sparse.csr_matrix(bm.array(
    [[1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 1., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 1., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 1., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 4., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 1., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0., 1., 0., 0.],
     [0., 0., 0., 0., 0., 0., 0., 1., 0.],
     [0., 0., 0., 0., 0., 0., 0., 0., 1.]]
))
A_after_none = CSRTensor(A_after_none_scipy.indptr, A_after_none_scipy.indices, A_after_none_scipy.data, A_after_none_scipy.shape)

f_after_none = bm.array([1., -1., 1., -1., 15.7392088, -1., 1., -1., 1.])

A_after_bd1_scipy = sparse.csr_matrix(bm.array(
    [[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  4., -1.,  0., -1.,  0.,  0.],
     [ 0.,  0.,  0., -1.,  4., -1.,  0., -1.,  0.],
     [ 0.,  0.,  0.,  0., -1.,  4.,  0.,  0., -1.],
     [ 0.,  0.,  0., -1.,  0.,  0.,  4., -1.,  0.],
     [ 0.,  0.,  0.,  0., -1.,  0., -1.,  4., -1.],
     [ 0.,  0.,  0.,  0.,  0., -1.,  0., -1.,  4.]]
))

A_after_bd1 = CSRTensor(A_after_bd1_scipy.indptr, A_after_bd1_scipy.indices, A_after_bd1_scipy.data, A_after_bd1_scipy.shape)

f_after_bd1 = bm.array([  1., -1.,  1., -18.7392088,  18.7392088, -18.7392088,
                          19.7392088, -19.7392088,  19.7392088])