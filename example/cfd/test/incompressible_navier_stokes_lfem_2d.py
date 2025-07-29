from fealpy.backend import backend_manager as bm
from fealpy.cfd.model.test.model_manager import CFDTestModelManager
from fealpy.cfd.incompressible_navier_stokes_lfem_2d_model import IncompressibleNSLFEM2DModel
import argparse


parser = argparse.ArgumentParser(description=
    """
    Solve elliptic equations using the lowest order Raviart-Thomas element and piecewise constant mixed finite element method.
    """)

parser.add_argument('--backend',
    default='numpy', type=str,
    help="Default backend is numpy. You can also choose pytorch, jax, tensorflow, etc.")

parser.add_argument('--GD',
    default='2d', type=str,
    help="Geometry dimension, default is 2d. You can also choose 3d.")
    
parser.add_argument('--pde',
    default=1, type=str,
    help="Name of the PDE model, default is sinsin")

parser.add_argument('--rho',
    default=1.0, type=float,
    help="Density of the fluid, default is 1.0")

parser.add_argument('--mu',
    default=1.0, type=float,
    help="Viscosity of the fluid, default is 1.0")

parser.add_argument('--init_mesh',
    default='tri', type=str,
    help="Type of initial mesh, default is tri")

parser.add_argument('--nx',
    default=8, type=int,
    help="Number of divisions in the x direction, default is 8")

parser.add_argument('--ny',
    default=8, type=int,
    help="Number of divisions in the y direction, default is 8")

parser.add_argument('--nz',
    default=8, type=int,
    help="Number of divisions in the z direction, default is 8 (only for 3D problems)")

parser.add_argument('--T0',
    default=0.0, type=float,
    help="Initial time, default is 0.0")

parser.add_argument('--T1',
    default=0.5, type=float,
    help="Final time, default is 0.5")

parser.add_argument('--nt',
    default=100, type=int,
    help="Number of time steps, default is 1000")

parser.add_argument('--method',
    default='Newton', type=str,
    help="Method for solving the PDE, default is Newton, options are Newton, Ossen, Stokes")

parser.add_argument('--solve',
    default='direct', type=str,
    help="Type of solver, default is direct, options are direct, iterative")

parser.add_argument('--apply_bc',
    default='dirichlet',type=str,
    help="Type of boundary condition application, default is dirichlet, options are dirichlet, neumann, cylinder, None")

parser.add_argument('--run',
    default='uniform_refine', type=str,
    help="Type of refinement strategy, default is uniform_refine")

parser.add_argument('--maxit',
    default=5, type=int,
    help="Maximum number of iterations for the solver, default is 5")

parser.add_argument('--maxstep',
    default=10, type=int,
    help="Maximum number of steps for the refinement, default is 10")

parser.add_argument('--tol',
    default=1e-10, type=float,
    help="Tolerance for the solver, default is 1e-10")

# 解析参数
options = vars(parser.parse_args())

bm.set_backend(options['backend'])
# pde = FromSympy(rho=options['rho'], mu=options['mu'])
bm.set_backend(options['backend'])
manager = CFDTestModelManager('incompressible_navier_stokes')
pde = manager.get_example(options['pde'], **options)
model = IncompressibleNSLFEM2DModel(pde, options = options)