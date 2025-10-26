from fealpy.backend import backend_manager as bm
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.cfd.stationary_incompressible_stokes_lfem_model import StationaryIncompressibleStokesLFEMModel
from fealpy.cfd.equation import StationaryIncompressibleNS
from fealpy.cfd.model import CFDPDEModelManager
from fealpy.mesher.dld_mesher import DLDMesher
import matplotlib.pyplot as plt
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
    default = 2, type=str,
    help="Name of the PDE model, default is sinsin")

parser.add_argument('--rho',
    default=1.0, type=float,
    help="Density of the fluid, default is 1.0")

parser.add_argument('--mu',
    default=1e-3, type=float,
    help="Viscosity of the fluid, default is 1.0")

parser.add_argument('--init_mesh',
    default = 'tri', type=str,
    help="Type of initial mesh, default is tri")

parser.add_argument('--bar',
    default = [0.0, 1, 0.0, 5], type=int,
    help="Computational domain [xmin, xmax, ymin, ymax]. Default: [0.0, 3, 0.0, 0.41].")

parser.add_argument('--m',
    default = 6, type=int,
    help="Radius of the circles, default is 0.029.")

parser.add_argument('--n',
    default = 8, type=int,
    help="Radius of the circles, default is 0.029.")

parser.add_argument('--theta',
    default = bm.pi / 10, type=float,
    help="Radius of the circles, default is 0.029.")

parser.add_argument('--radius',
    default = 0.1, type=float,
    help="Radius of the circles, default is 0.029.")

parser.add_argument('--T',
    default = 3, type=int,
    help="Radius of the circles, default is 0.029.")

parser.add_argument('--lc',
    default = 1e-1, type=float,
    help="Target mesh element size (characteristic length). Default: 0.01.")

parser.add_argument('--hole_lc',
    default = 8e-3, type=float,
    help="Mesh size, default is 0.006")

parser.add_argument('--return_project_edge',
    default='True', type=str,
    help='Whether to generate mesh, default is True')

parser.add_argument('--return_mesh',
    default='True', type=str,
    help='Whether to generate mesh, default is True')

parser.add_argument('--show_figure',
    default='True', type=str,
    help='Whether to show figure in Gmsh, default is True')

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
    default='main', type=str,
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

mesh = pde.init_mesh()


fig = plt.figure()
axes = fig.add_subplot(111)
mesh.add_plot(axes)
mesh.find_node(axes, 
               showindex=True, color='r', fontsize='2')
# mesh.find_cell(axes, showindex=True, fontsize='35')
plt.show()


model = StationaryIncompressibleNSLFEMModel(pde=pde, mesh = mesh, options = options)
uh, ph = model.run()
# model.__str__()


# 可视化
mesh = model.mesh
uh = uh.reshape(2, -1).T
points_u = model.fem.uspace.interpolation_points()
points_p = model.fem.pspace.interpolation_points()

plt.figure(figsize=(18, 15))
plt.tricontourf(points_u[:, 0], points_u[:, 1], mesh.cell, uh[..., 0], levels = 50, cmap='viridis')
plt.colorbar(label = 'uh0')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Velocity uh0')
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 15))
plt.tricontourf(points_u[:, 0], points_u[:, 1], mesh.cell, uh[..., 1], levels = 50, cmap='viridis')
plt.colorbar(label = 'uh1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Velocity uh1')
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 15))
plt.tricontourf(points_p[:, 0], points_p[:, 1], mesh.cell, ph, levels = 50, cmap='viridis')
plt.colorbar(label = 'ph')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Pressure ph')
plt.grid(True)
plt.show()

from scipy.interpolate import griddata
x, y = points_u[:, 0], points_u[:, 1]

# 创建规则网格
xi = bm.linspace(x.min(), x.max(), 200)
yi = bm.linspace(y.min(), y.max(), 200)
X, Y = bm.meshgrid(xi, yi)

# 插值速度分量到规则网格上
Ui = griddata(points_u, uh[..., 0], (X, Y), method='linear')
Vi = griddata(points_u, uh[..., 1], (X, Y), method='linear')

# 绘制流线图
plt.figure(figsize=(8, 6))
plt.streamplot(X, Y, Ui, Vi, density=1.5, color=bm.sqrt(Ui**2 + Vi**2), cmap='viridis')
plt.colorbar(label='|u|')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Streamlines of velocity field')
plt.axis('equal')
plt.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.gca()
mesh.add_plot(ax)
plt.axis("equal")
plt.show()

