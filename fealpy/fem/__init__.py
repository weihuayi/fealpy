
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

from .TrussStructureIntegrator import TrussStructureIntegrator
from .DiffusionIntegrator import DiffusionIntegrator
from .ConvectionIntegrator import ConvectionIntegrator
from .LinearElasticityOperatorIntegrator import LinearElasticityOperatorIntegrator

from .ProvidesSymmetricTangentOperatorIntegrator import ProvidesSymmetricTangentOperatorIntegrator


from .DirichletBC import DirichletBC

