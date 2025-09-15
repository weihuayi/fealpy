from fealpy.backend import backend_manager as bm
from fealpy.cfd.model import CFDPDEModelManager
from fealpy.cfd.incompressible_navier_stokes_lfem_2d_model import IncompressibleNSLFEM2DModel
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
    default=1.0, type=float,
    help="Viscosity of the fluid, default is 1.0")

parser.add_argument('--T0',
    default=0.0, type=float,
    help="Initial time, default is 0.0")

parser.add_argument('--T1',
    default=6.0, type=float,
    help="Final time, default is 0.5")

parser.add_argument('--nt',
    default=24000, type=int,
    help="Number of time steps, default is 1000")

parser.add_argument('--init_mesh',
    default = 'tri', type=str,
    help="Type of initial mesh, default is tri")

parser.add_argument('--box',
    default = [0.0, 2.2, 0.0, 0.41], type=int,
    help="Computational domain [xmin, xmax, ymin, ymax]. Default: [0.0, 2.2, 0.0, 0.41].")

parser.add_argument('--center',
    default = (0.2, 0.2), type=float,
    help="Center of the first circle, default is (0.2, 0.2).")

parser.add_argument('--radius',
    default = 0.05, type=int,
    help="Radius of the circles, default is 0.05.")

parser.add_argument('--n_circle',
    default = 400, type=int,
    help="Number of divisions in the circle, default is 60")

parser.add_argument('--lc',
    default = 0.01, type=float,
    help="Target mesh element size (characteristic length). Default: 0.01.")

parser.add_argument('--method',
    default='IPCS', type=str,
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
    default='main_cylinder', type=str,
    help="Type of refinement strategy, default is uniform_refine")

parser.add_argument('--maxit',
    default=5, type=int,
    help="Maximum number of iterations for the solver, default is 5")

parser.add_argument('--maxstep',
    default=10, type=int,
    help="Maximum number of steps for the refinement, default is 1000")

parser.add_argument('--tol',
    default=1e-10, type=float,
    help="Tolerance for the solver, default is 1e-10")

# 解析参数
options = vars(parser.parse_args())

from fealpy.old.pde.navier_stokes_equation_2d import FlowPastCylinder
bm.set_backend(options['backend'])
# bm.set_default_device('cpu')
manager = CFDPDEModelManager('incompressible_navier_stokes')
pde = manager.get_example(options['pde'], **options)
mesh = pde.init_mesh()
model = IncompressibleNSLFEM2DModel(pde=pde, mesh = mesh, options = options)
model.equation.set_constitutive(1)
model.equation.set_coefficient('viscosity', pde.mu)
uh, ph = model.run()
cd = model.cd
cl = model.cl
delta_p = model.delta_p
x = bm.linspace(0.0, 5.0, model.timeline.NL)
# model.__str__()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x[16000:20000], cd[15999:19999], marker=None, linestyle='-', color='black')
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Drag coefficient', fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x[16000:20000], cl[15999:19999], marker=None, linestyle='-', color='black')
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Lift coefficient', fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x[16000:20000], delta_p[15999:19999], marker=None, linestyle='-', color='black')
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Pressure difference', fontsize=14)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(x[16000:], cd[15999:], marker=None, linestyle='-', color='black')
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Drag coefficient', fontsize=14)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(x[16000:], cl[15999:], marker=None, linestyle='-', color='black')
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Lift coefficient', fontsize=14)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x[16000:], delta_p[15999:], marker=None, linestyle='-', color='black')
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Pressure difference', fontsize=14)
plt.show()
