
'''
femmodel

This module provide many fem model 

'''

from .bilinear_form import BilinearForm
from .linear_form import LinearForm

# Integrator for scalar case
from .scalar_mass_integrator import ScalarMassIntegrator
from .scalar_diffusion_integrator import ScalarDiffusionIntegrator
from .scalar_source_integrator import ScalarSourceIntegrator

# Integrator for vector case
from .vector_source_integrator import VectorSourceIntegrator
from .vector_diffusion_integrator import VectorDiffusionIntegrator

from .truss_structure_integrator import TrussStructureIntegrator
from .diffusion_integrator import DiffusionIntegrator
from .convection_integrator import ConvectionIntegrator
from .linear_elasticity_operator_integrator import LinearElasticityOperatorIntegrator

from .provides_symmetric_tangent_operator_integrator import ProvidesSymmetricTangentOperatorIntegrator

from .vector_neumann_bc_integrator import VectorNeumannBCIntegrator
from .scalar_neumann_bc_integrator import ScalarNeumannBCIntegrator

from .dirichlet_bc import DirichletBC

