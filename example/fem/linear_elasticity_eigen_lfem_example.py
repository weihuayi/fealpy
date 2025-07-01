
import argparse
from fealpy.backend import bm

from fealpy.model import PDEDataManager
from fealpy.fem import LinearElasticityEigenLFEMModel
from fealpy.mesh import TriangleMesh

pde = PDEDataManager('linear_elasticity').get_example('boxpoly')

mesh = pde.init_mesh()
model = LinearElasticityEigenLFEMModel(mesh)
