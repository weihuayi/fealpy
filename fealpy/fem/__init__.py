
'''
femmodel

This module provide many fem model 

'''

from .BilinearForm import BilinearForm
from .LinearForm import LinearForm

# Integrator for scalar case
from .ScalarMassIntegrator import ScalarMassIntegrator
from .ScalarDiffusionIntegrator import ScalarDiffusionIntegrator
from .ScalarSourceIntegrator import ScalarSourceIntegrator

# Integrator for vector case
from .VectorSourceIntegrator import VectorSourceIntegrator
from .VectorDiffusionIntegrator import VectorDiffusionIntegrator

from .TrussStructureIntegrator import TrussStructureIntegrator
from .DiffusionIntegrator import DiffusionIntegrator
from .ConvectionIntegrator import ConvectionIntegrator
from .LinearElasticityOperatorIntegrator import LinearElasticityOperatorIntegrator

from .ProvidesSymmetricTangentOperatorIntegrator import ProvidesSymmetricTangentOperatorIntegrator

from .VectorNeumannBoundaryIntegrator import VectorNeumannBoundaryIntegrator
from .ScalarNeumannBoundaryIntegrator import ScalarNeumannBoundaryIntegrator

from .DirichletBC import DirichletBC

