from fealpy.backend import backend_manager as bm
from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.cfd.model.test.stationary_incompressible_navier_stokes.stationary_incompressible_navier_stokes_2d import FromSympy
from fealpy.cfd.model.test import CFDTestModelManager
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
    default = 8, type = int,
    help = "Number of divisions in the x direction, default is 8")

parser.add_argument('--ny',
    default = 8, type = int,
    help = "Number of divisions in the y direction, default is 8")

parser.add_argument('--nz',
    default = 8, type = int,
    help = "Number of divisions in the z direction, default is 8 (only for 3D problems)")

parser.add_argument('--method',
    default = 'Newton', type = str,
    help = "Method for solving the PDE, default is Newton, options are Newton, Ossen, Stokes")

parser.add_argument('--solve',
    default = 'direct', type = str,
    help = "Type of solver, default is direct, options are direct, iterative")

parser.add_argument('--run',
    default = 'uniform_refine', type = str,
    help = "Type of refinement strategy, default is uniform_refine")

parser.add_argument('--maxit',
    default = 1, type = int,
    help = "Maximum number of iterations for the solver, default is 5")

parser.add_argument('--maxstep',
    default = 1000, type = int,
    help = "Maximum number of steps for the refinement, default is 1000")

parser.add_argument('--tol',
    default = 1e-10, type = float,
    help = "Tolerance for the solver, default is 1e-10")

# 解析参数
options = vars(parser.parse_args())

bm.set_backend(options['backend'])
manager = CFDTestModelManager('stationary_incompressible_navier_stokes')
pde = manager.get_example(options['pde'], **options)
# print(pde.__str__())
equation = StationaryIncompressibleNS(pde=pde)

model = StationaryIncompressibleNSLFEMModel(equation, options)
# print(model.fem.params)
# print(model.equation)
# model.__str__()
