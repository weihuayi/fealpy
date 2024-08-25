from fealpy.experimental.backend import backend_manager as bm

bm.set_backend('numpy')
# bm.set_backend('pytorch')
# bm.set_backend('jax')

from fealpy.experimental.mesh import UniformMesh2d
from fealpy.mesh import UniformMesh2d as UniformMesh2d_old

from fealpy.experimental.fem import LinearElasticityIntegrator, \
                                    BilinearForm, LinearForm, \
                                    VectorSourceIntegrator

from fealpy.fem import LinearElasticityOperatorIntegrator as LinearElasticityIntegrator_old
from fealpy.fem import VectorSourceIntegrator as VectorSourceIntegrator_old
from fealpy.fem import BilinearForm as BilinearForm_old
from fealpy.fem import LinearForm as LinearForm_old
from fealpy.fem import DirichletBC as DirichletBC_old

from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.functionspace import LagrangeFESpace as LagrangeFESpace_old

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.solver import cg
from fealpy.experimental.fem import DirichletBC as DBC
from fealpy.experimental.sparse import COOTensor


extent = [0, 2, 0, 2]
h = [1.5, 1]
origin = [0, 0]
mesh = UniformMesh2d(extent, h, origin)

ip2 = mesh.interpolation_points(p=2)
edge = mesh.entity('edge')
cell = mesh.entity('cell')
edge2ipoint = mesh.edge_to_ipoint(p=2)
cell2ipoint = mesh.cell_to_ipoint(p=2)
edgenorm = mesh.edge_normal()
edgeunitnorrm = mesh.edge_unit_normal()
cellnorm = mesh.cell_normal()


import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
# mesh.find_node(axes, node=ip2, showindex=True)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()