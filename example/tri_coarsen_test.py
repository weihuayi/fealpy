import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import Exp 
from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from fealpy.mesh.Tritree import Tritree


theta = 0.5
beta = 0.5

pde = Exp()
mesh = pde.init_mesh(n=2)
tmesh = Tritree(mesh.node, mesh.ds.cell)
pmesh = tmesh.to_conformmesh()
femspace = LagrangeFiniteElementSpace(pmesh, p=1) 
mesh = femspace.mesh
uI = femspace.interpolation(pde.solution) # calculate the value of the interpolation point
cellmeasure = mesh.entity_measure('cell')

NC = mesh.number_of_cells()
estimator = np.zeros(NC, dtype=mesh.ftype) 
bc = np.array([1/3, 1/3, 1/3], dtype=np.float)
grad_uI = uI.grad_value(bc)
estimator[:] = np.sqrt(np.sum(grad_uI**2, axis=1)*cellmeasure) # compute estimator

isMarkedCell = tmesh.refine_marker(estimator, theta, method='L2')
tmesh.refine(isMarkedCell)
pmesh = tmesh.to_conformmesh()

#isMarkedCell = tmesh.coarsen_marker(estimator, beta, method='COARSEN')
#tmesh.coarsen(isMarkedCell)

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
#tmesh.find_node(axes, showindex=True)
#tmesh.find_cell(axes, showindex=True)
#tmesh.find_edge(axes, showindex=True)
plt.show()

