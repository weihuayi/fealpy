"""The FEM Module"""

### Forms and bases
from .integrator import *
from .bilinear_form import BilinearForm
from .linear_form import LinearForm
from .nonlinear_form import NonlinearForm
from .block_form import BlockForm
from .linear_block_form import LinearBlockForm

### Cell Operator
from .scalar_diffusion_integrator import ScalarDiffusionIntegrator
from .scalar_nonlinear_diffusion_integrator import ScalarNonlinearDiffusionIntegrator
from .scalar_mass_integrator import ScalarMassIntegrator
from .scalar_nonlinear_mass_integrator import ScalarNonlinearMassIntegrator
from .scalar_convection_integrator import ScalarConvectionIntegrator
from .linear_elastic_integrator import LinearElasticIntegrator
from .press_work_integrator import PressWorkIntegrator, PressWorkIntegratorX, PressWorkIntegratorY
from .vector_mass_integrator import VectorMassIntegrator
from .curlcurl_integrator import CurlCurlIntegrator
from .nonlinear_elastic_integrator import NonlinearElasticIntegrator
from .div_integrator import DivIntegrator
from .viscous_work_integrator import ViscousWorkIntegrator
from .scalar_biharmonic_integrator import ScalarBiharmonicIntegrator

### Cell Source
from .cell_source_integrator import CellSourceIntegrator
SourceIntegrator = CellSourceIntegrator

from .grad_source_integrator import GradSourceIntegrator
from .scalar_source_integrator import ScalarSourceIntegrator
from .vector_source_integrator import VectorSourceIntegrator

### Face Operator
from .scalar_robin_bc_integrator import ScalarRobinBCIntegrator
from .face_mass_integrator import BoundaryFaceMassIntegrator, InterFaceMassIntegrator
from .fluid_boundary_friction_integrator import FluidBoundaryFrictionIntegrator
from .scalar_interior_penalty_integrator import ScalarInteriorPenaltyIntegrator
from .press_work_integrator import BoundaryPressWorkIntegrator

### Face Source
from .face_source_integrator import BoundaryFaceSourceIntegrator, InterFaceSourceIntegrator
ScalarNeumannBCIntegrator = BoundaryFaceSourceIntegrator
ScalarRobinSourceIntegrator = BoundaryFaceSourceIntegrator

### Dirichlet BC
from .dirichlet_bc import DirichletBC
from .dirichlet_bc_operator import DirichletBCOperator

### recovery estimate
from .recovery_alg import RecoveryAlg

### Other
from .nonlinear_wrapper import NonlinearWrapperInt


### solver
from .poisson_lfem_solver import PoissonLFEMSolver
from .linear_elasticity_lfem_solver import LinearElasticityLFEMSolver
from .level_set_lfem_model import LevelSetLFEMModel, LevelSetReinitModel
from .interface_poisson_lfem_model import InterfacePoissonLFEMModel
