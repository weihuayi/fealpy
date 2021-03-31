import platform

from .solve import solve, active_set_solver
from .amg import AMGSolver
from .matlab_solver import MatlabSolver

#try:
#    from .petsc_solver import PETScSolver
#except ImportError:
#    print('I do not find petsc and petsc4py installed on this system!, so you can not use it')

from .fast_solver import HighOrderLagrangeFEMFastSolver
from .fast_solver import SaddlePointFastSolver
from .fast_solver import LinearElasticityLFEMFastSolver 

from .LinearElasticityRLFEMFastSolver import LinearElasticityRLFEMFastSolver
