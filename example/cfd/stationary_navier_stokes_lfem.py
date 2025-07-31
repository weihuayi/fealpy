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
    default = [0.0, 2.2, 0.0, 0.41], type=int,
    help="Number of divisions in the x direction, default is 8")

parser.add_argument('--center',
    default = (0.5, 0.205), type=int,
    help="Number of divisions in the y direction, default is 8")

parser.add_argument('--radius',
    default = 0.05, type=int,
    help="Number of divisions in the z direction, default is 8 (only for 3D problems)")

parser.add_argument('--n_circle',
    default = 100, type=int,
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
model = StationaryIncompressibleNSLFEMModel(pde=pde, options = options)







# 可视化
mesh = pde.mesh
uh, ph = model.uh1, model.ph1
uh = uh.reshape(2, -1).T
points = model.fem.uspace.interpolation_points()
print(model.fem.uspace.interpolation_points())
triang = tri.Triangulation(points[:, 0], points[:, 1])

plt.figure(figsize=(18, 3))
plt.tricontourf(points[:, 0], points[:, 1], mesh.cell, uh[..., 0], levels = 50, cmap='viridis')
plt.colorbar(label = 'uh')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Velocity uh0')
plt.grid(True)
plt.show()

plt.figure(figsize=(18, 3))
plt.tricontourf(points[:, 0], points[:, 1], mesh.cell, uh[..., 1], levels = 50, cmap='viridis')
plt.colorbar(label = 'uh')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Velocity uh1')
plt.grid(True)
plt.show()

from scipy.interpolate import griddata
x, y = points[:, 0], points[:, 1]

# 创建规则网格
xi = bm.linspace(x.min(), x.max(), 200)
yi = bm.linspace(y.min(), y.max(), 200)
X, Y = bm.meshgrid(xi, yi)

# 插值速度分量到规则网格上
Ui = griddata(points, uh[..., 0], (X, Y), method='linear')
Vi = griddata(points, uh[..., 1], (X, Y), method='linear')

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
# mesh.find_node(ax, showindex=True)
# mesh.find_edge(ax, showindex=True)
# mesh.find_cell(ax, showindex=True)
plt.axis("equal")
plt.show()

