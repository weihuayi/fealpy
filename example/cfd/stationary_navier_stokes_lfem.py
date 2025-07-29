from fealpy.backend import backend_manager as bm
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.cfd.model import CFDPDEModelManager
from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
from fealpy.functionspace import Function
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import argparse

## 参数解析
parser = argparse.ArgumentParser(description=
    """
    Solve elliptic equations using the lowest order Raviart-Thomas element and piecewise constant mixed finite element method.
    """)

parser.add_argument('--backend',
    default='numpy', type=str,
    help="Default backend is numpy. You can also choose pytorch, jax, tensorflow, etc.")

parser.add_argument('--pde',
    default = 1, type=str,
    help="Name of the PDE model, default is sinsin")

parser.add_argument('--rho',
    default=1.0, type=float,
    help="Density of the fluid, default is 1.0")

parser.add_argument('--mu',
    default=1.0, type=float,
    help="Viscosity of the fluid, default is 1.0")

parser.add_argument('--init_mesh',
    default = 'tri', type=str,
    help="Type of initial mesh, default is tri")

parser.add_argument('--box',
    default = [0.0, 5.0, 0.0, 1.0], type=int,
    help="Number of divisions in the x direction, default is 8")

parser.add_argument('--center',
    default = (1.0, 0.5), type=int,
    help="Number of divisions in the y direction, default is 8")

parser.add_argument('--radius',
    default = 0.5, type=int,
    help="Number of divisions in the z direction, default is 8 (only for 3D problems)")

parser.add_argument('--n_circle',
    default = 60, type=int,
    help="Number of divisions in the circle, default is 60")

parser.add_argument('--h',
    default = 0.05, type=float,
    help="Mesh size, default is 0.05")

parser.add_argument('--method',
    default='Newton', type=str,
    help="Method for solving the PDE, default is Newton, options are Newton, Ossen, Stokes")

parser.add_argument('--solve',
    default='direct', type=str,
    help="Type of solver, default is direct, options are direct, iterative")

parser.add_argument('--apply_bc',
    default='cylinder', type=str,
    help="Type of boundary condition application, default is dirichlet, options are dirichlet, neumann, cylinder, None")

parser.add_argument('--postprocess',
    default='res', type=str,
    help="Post-processing method, default is error, options are error, plot")

parser.add_argument('--run',
    default='one_step', type=str,
    help="Type of refinement strategy, default is uniform_refine")

parser.add_argument('--maxit',
    default=5, type=int,
    help="Maximum number of iterations for the solver, default is 5")

parser.add_argument('--maxstep',
    default=1000, type=int,
    help="Maximum number of steps for the refinement, default is 1000")

parser.add_argument('--tol',
    default=1e-10, type=float,
    help="Tolerance for the solver, default is 1e-10")

# 解析参数
options = vars(parser.parse_args())

bm.set_backend(options['backend'])
manager = CFDPDEModelManager('stationary_incompressible_navier_stokes')
pde = manager.get_example(options['pde'], **options)
model = StationaryIncompressibleNSLFEMModel(pde=pde, options = options)
# print(model.fem.params)
# model.plot(uh = model.uh1, ph = model.ph1)

# 可视化
mesh = pde.mesh
uh, ph = model.uh1, model.ph1
ugdof = model.fem.uspace.number_of_global_dofs()
points = mesh.entity_barycenter('node')
points = points.astype(bm.float64)
triang = tri.Triangulation(points[:, 0], points[:, 1])
uh = bm.sum(uh(points), axis = 0).T
# print(points)
print(uh)


exit()
plt.figure(figsize=(15, 3))
# plt.streamplot(points[:, 0], points[:, 1], uh[0], uh[1], color=bm.sqrt(uh[0]**2 + uh[1]**2), cmap='viridis', density=1.5)
# contour = plt.tricontourf(points[:, 0], points[:, 1], uh[0], triangles = mesh.cell, levels = 50, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Velocity u1')
plt.grid(True)
plt.show()

fig = plt.figure()
ax = fig.gca()
mesh.add_plot(ax)
# mesh.find_node(ax, showindex=True)
# mesh.find_edge(ax, showindex=True)
# mesh.find_cell(ax, showindex=True)
plt.axis("equal")
plt.show()

