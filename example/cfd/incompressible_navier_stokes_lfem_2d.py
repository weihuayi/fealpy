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
    default=8.0, type=float,
    help="Final time, default is 0.5")

parser.add_argument('--nt',
    default=8000, type=int,
    help="Number of time steps, default is 1000")

parser.add_argument('--init_mesh',
    default = 'tri', type=str,
    help="Type of initial mesh, default is tri")

parser.add_argument('--box',
    default = [0.0, 2.2, 0.0, 0.41], type=int,
    help="Computational domain [xmin, xmax, ymin, ymax]. Default: [0.0, 3, 0.0, 0.41].")

parser.add_argument('--center',
    default = (0.2, 0.2), type=float,
    help="Center of the first circle, default is (0.1, 0.05).")

parser.add_argument('--radius',
    default = 0.05, type=int,
    help="Radius of the circles, default is 0.029.")

parser.add_argument('--n_circle',
    default = 200, type=int,
    help="Number of divisions in the circle, default is 60")

parser.add_argument('--lc',
    default = 0.05, type=float,
    help="Target mesh element size (characteristic length). Default: 0.01.")

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
    default=10, type=int,
    help="Maximum number of steps for the refinement, default is 1000")

parser.add_argument('--tol',
    default=1e-10, type=float,
    help="Tolerance for the solver, default is 1e-10")

# 解析参数
options = vars(parser.parse_args())

bm.set_backend(options['backend'])
manager = CFDPDEModelManager('incompressible_navier_stokes')
pde = manager.get_example(options['pde'], **options)
mesh = pde.init_mesh()
model = IncompressibleNSLFEM2DModel(pde=pde, mesh = mesh, options = options)
uh, ph = model.run()
# model.__str__()