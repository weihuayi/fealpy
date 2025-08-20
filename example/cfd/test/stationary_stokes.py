from fealpy.backend import backend_manager as bm
from fealpy.cfd.stationary_incompressible_stokes_lfem_model import StationaryIncompressibleStokesLFEMModel
from fealpy.cfd.equation import StationaryIncompressibleNS
from fealpy.cfd.model import CFDTestModelManager
import matplotlib.pyplot as plt
import argparse

## 参数解析
parser = argparse.ArgumentParser(description=
    """
    Solve elliptic equations using the lowest order Raviart-Thomas element and piecewise constant mixed finite element method.
    """)

parser.add_argument('--backend',
    default = 'numpy', type = str,
    help = "Default backend is numpy. You can also choose pytorch, jax, tensorflow, etc.")
    
parser.add_argument('--pde',
    default = 1, type = str,
    help = "Name of the PDE model, default is exp0001")

parser.add_argument('--init_mesh',
    default = 'tri', type = str,
    help = "Type of initial mesh, default is tri")

parser.add_argument('--nx',
    default = 4, type = int,
    help = "Number of divisions in the x direction, default is 8")

parser.add_argument('--ny',
    default = 4, type = int,
    help = "Number of divisions in the y direction, default is 8")

parser.add_argument('--nz',
    default = 2, type = int,
    help = "Number of divisions in the z direction, default is 8 (only for 3D problems)")

parser.add_argument('--method',
    default = 'Stokes', type = str,
    help = "Method for solving the PDE, default is Newton, options are Newton, Ossen, Stokes")

parser.add_argument('--solve',
    default = 'direct', type = str,
    help = "Type of solver, default is direct, options are direct, iterative")

parser.add_argument('--apply_bc',
    default = 'dirichlet', type = str,
    help = "Type of boundary condition application, default is dirichlet, options are dirichlet, neumann, cylinder, None")

parser.add_argument('--run',
    default = 'uniform_refine', type = str,
    help = "Type of refinement strategy, default is uniform_refine")

parser.add_argument('--maxit',
    default = 5, type = int,
    help = "Maximum number of iterations for the solver, default is 5")

parser.add_argument('--maxstep',
    default = 10, type = int,
    help = "Maximum number of steps for the refinement, default is 1000")

parser.add_argument('--tol',
    default = 1e-10, type = float,
    help = "Tolerance for the solver, default is 1e-10")

# 解析参数
options = vars(parser.parse_args())

bm.set_backend(options['backend'])
manager = CFDTestModelManager('stationary_stokes')
pde = manager.get_example(options['pde'], **options)
mesh = pde.mesh
model = StationaryIncompressibleStokesLFEMModel(pde=pde, mesh = mesh, options = options)
uh, ph = model.run()
model.__str__()


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

# from scipy.interpolate import griddata
# x, y = points_u[:, 0], points_u[:, 1]

# # 创建规则网格
# xi = bm.linspace(x.min(), x.max(), 200)
# yi = bm.linspace(y.min(), y.max(), 200)
# X, Y = bm.meshgrid(xi, yi)

# # 插值速度分量到规则网格上
# Ui = griddata(points_u, uh[..., 0], (X, Y), method='linear')
# Vi = griddata(points_u, uh[..., 1], (X, Y), method='linear')

# # 绘制流线图
# plt.figure(figsize=(8, 6))
# plt.streamplot(X, Y, Ui, Vi, density=1.5, color=bm.sqrt(Ui**2 + Vi**2), cmap='viridis')
# plt.colorbar(label='|u|')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Streamlines of velocity field')
# plt.axis('equal')
# plt.tight_layout()
# plt.show()

fig = plt.figure()
ax = fig.gca()
mesh.add_plot(ax)
plt.axis("equal")
plt.show()

