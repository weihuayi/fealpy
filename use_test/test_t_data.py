from fealpy.mesh import TriangleMesh
from fealpy.backend import backend_manager as bm
from fealpy.sparse.coo_tensor import COOTensor
import matplotlib.pyplot as plt

node = bm.tensor([[0,0],[1,0],[0,1]],dtype=bm.float64)
cell=bm.tensor([[0,1,2]],dtype=bm.int32)
mesh = TriangleMesh(node,cell)
mesh.uniform_refine()
mesh.uniform_refine()
numcell = mesh.number_of_cells()
cell_new = bm.ones(numcell)
isMarkedCell = (cell_new == 1.0).astype(bm.bool)
mesh.coarsen(isMarkedCell=isMarkedCell)
n=2

# A = mesh.uniform_refine(n,returnim=True)
# def linear_function(coodrs):
#     return 2*coodrs[:,0] + 2*coodrs[:,1]+2
# excat_value = linear_function(mesh.node)
# val = linear_function(node)
# test_value = val
# for i in range(n):
#     print(A[i])
#     test_value = A[i].dot(test_value)
# print(excat_value)
# print(test_value)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
plt.show()