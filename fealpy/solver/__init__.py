import platform

from .solve import solve, active_set_solver
from .gamg_solver import GAMGSolver
from .cupy_solver import CupySolver

try:
    from .matlab_solver import MatlabSolver
except ImportError:
    print('I do not find matlab installed on this system!, so you can not use it')

#try:
#    from .petsc_solver import PETScSolver
#except ImportError:
#    print('I do not find petsc and petsc4py installed on this system!, so you can not use it')

from .fast_solver import HighOrderLagrangeFEMFastSolver
from .fast_solver import SaddlePointFastSolver
from .fast_solver import LinearElasticityLFEMFastSolver 
from .fast_solver import LevelSetFEMFastSolver 

from .LinearElasticityRLFEMFastSolver import LinearElasticityRLFEMFastSolver
