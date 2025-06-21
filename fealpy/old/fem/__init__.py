
'''
femmodel

This module provide many fem model 

'''

from .bilinear_form import BilinearForm
from .mixed_bilinear_form import MixedBilinearForm
from .linear_form import LinearForm
from .nonlinear_form import NonlinearForm

# Domain integrator for scalar case
from .scalar_diffusion_integrator import ScalarDiffusionIntegrator
ScalarLaplaceIntegrator = ScalarDiffusionIntegrator 
from .scalar_convection_integrator import ScalarConvectionIntegrator
from .scalar_mass_integrator import ScalarMassIntegrator
from .scalar_source_integrator import ScalarSourceIntegrator

from .scalar_pgls_convection_integrator import ScalarPGLSConvectionIntegrator

from .scalar_biharmonic_integrator import ScalarBiharmonicIntegrator
from .scalar_interior_penalty_integrator import ScalarInteriorPenaltyIntegrator


# Boundary integrator for scalar case
# <kappa u, v>
from .scalar_robin_boundary_integrator import ScalarRobinBoundaryIntegrator
# <g, v>
from .scalar_boundary_source_integrator import ScalarBoundarySourceIntegrator
from .scalar_interface_integrator import ScalarInterfaceIntegrator
# <g_N, v>
ScalarNeumannSourceIntegrator = ScalarBoundarySourceIntegrator
# <g_R, v>
ScalarRobinSourceIntegrator = ScalarBoundarySourceIntegrator

# Domain integrator for vector case
from .vector_diffusion_integrator import VectorDiffusionIntegrator
from .vector_mass_integrator import VectorMassIntegrator
from .vector_source_integrator import VectorSourceIntegrator
from .vector_epsilon_source_integrator import VectorEpsilonSourceIntegrator
from .linear_elasticity_operator_integrator import LinearElasticityOperatorIntegrator

# Boundary integrator for vector case
from .vector_boundary_source_integrator import VectorBoundarySourceIntegrator
VectorNeumannSourceIntegrator = VectorBoundarySourceIntegrator
VectorRobinSourceIntegrator = VectorBoundarySourceIntegrator


# others
from .truss_structure_integrator import TrussStructureIntegrator
from .beam_structure_integrator import EulerBernoulliCantileverBeamStructureIntegrator
from .beam_structure_integrator import EulerBernoulliBeamStructureIntegrator
from .diffusion_integrator import DiffusionIntegrator
from .vector_convection_integrator import VectorConvectionIntegrator
from .vector_viscous_work_integrator import VectorViscousWorkIntegrator
from .press_work_integrator import PressWorkIntegrator

from .provides_symmetric_tangent_operator_integrator import ProvidesSymmetricTangentOperatorIntegrator

from .vector_neumann_bc_integrator import VectorNeumannBCIntegrator
from .scalar_neumann_bc_integrator import ScalarNeumannBCIntegrator

from .dirichlet_bc import DirichletBC
from .recovery_alg import recovery_alg, LinearRecoveryAlg
from .fluid_boundary_friction_integrator import FluidBoundaryFrictionIntegrator



from .darcy_integrator import VectorDarcyIntegrator
