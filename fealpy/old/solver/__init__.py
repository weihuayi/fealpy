import platform

from .. import logger
from .solve import solve, active_set_solver
#from .gamg_solver import GAMGSolver

try:
    from .cupy_solver import CupySolver
except ImportError:
    logger.info("Can't import CupySolver! If you want to use, please install it and try again.")

try:
    from .matlab_solver import MatlabSolver
except ImportError:
    logger.info("Can't import MatlabSolver! If you want to use it, please install it and try again")

#try:
#    from .petsc_solver import PETScSolver
#except ImportError:
#    print('I do not find petsc and petsc4py installed on this system!, so you can not use it')

from .fast_solver import HighOrderLagrangeFEMFastSolver
from .fast_solver import SaddlePointFastSolver
from .fast_solver import LinearElasticityLFEMFastSolver 
from .fast_solver import LevelSetFEMFastSolver 

from .LinearElasticityRLFEMFastSolver import LinearElasticityRLFEMFastSolver

#from .gamg_solver import GAMGSolver
