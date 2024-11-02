import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from fealpy.quadrature.FEMeshIntegralAlg import FEMeshIntegralAlg
from fealpy.mesh import Tritree
from fealpy.mesh import TriangleMesh


def f1(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.exp(5*(x**2 + y**2))/np.exp(10)
    return val

def f2(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.exp(5*(x**2 + (y-1)**2))/np.exp(10)
    return val

theta = 0.35
node = np.array([
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1)], dtype=np.float)

cell = np.array([
    (1, 2, 0), 
    (3, 0, 2)], dtype=np.int)
mesh = TriangleMesh(node, cell)
mesh.uniform_refine(4)

integrator = mesh.integrator(3)

node = mesh.entity('node')
cell = mesh.entity('cell')
tmesh = Tritree(node, cell)

femspace = LagrangeFiniteElementSpace(mesh, p=1)

integralalg = FEMeshIntegralAlg(integrator, mesh)
qf = integrator
bcs, ws = qf.quadpts, qf.weights

u1 = femspace.interpolation(f1)
grad = femspace.grad_value(u1, bcs, cellidx=None)
eta1 = integralalg.L2_norm1(grad, celltype=True)

u2 = femspace.interpolation(f2)
grad = femspace.grad_value(u2, bcs, cellidx=None)
eta2 = integralalg.L2_norm1(grad, celltype=True)

isMarkedCell = tmesh.refine_marker(eta2, theta, "MAX")
tmesh.refine(isMarkedCell)
pmesh = tmesh.to_conformmesh()

fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)
plt.show()





















#fig = plt.figure()
#axes = fig.gca() 
#mesh.add_plot(axes, cellcolor=estimator.eta, showcolorbar=True)
#
#tmesh.adaptive_refine(estimator)
#mesh = estimator.mesh
#fig = plt.figure()
#axes = fig.gca() 
#mesh.add_plot(axes, cellcolor=estimator.eta, showcolorbar=True)
#
#
#femspace = LagrangeFiniteElementSpace(mesh, p=1)
#uI = femspace.interpolation(f2)
#estimator = Estimator(uI[:], mesh, 0.3, 0.5)
#
#tmesh.adaptive_coarsen(estimator)
#mesh = estimator.mesh
#fig = plt.figure()
#axes = fig.gca() 
#mesh.add_plot(axes, cellcolor=estimator.eta, showcolorbar=True)
#
#femspace = LagrangeFiniteElementSpace(mesh, p=1)
#uI = femspace.interpolation(f2)
#tmesh.adaptive_refine(estimator)
#mesh = estimator.mesh
#fig = plt.figure()
#axes = fig.gca() 
#mesh.add_plot(axes, cellcolor=estimator.eta, showcolorbar=True)
#
#plt.show()

