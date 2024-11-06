
import numpy as np
import sys

from fealpy.model.poisson_model_2d import CosCosData
from fealpy.vemmodel import PoissonVEMModel 
from fealpy.quadrature import QuadrangleQuadrature 
from fealpy.mesh import PolygonMesh
from fealpy.mesh.tree_data_structure import Quadtree

p = int(sys.argv[1])

model = CosCosData()
point = np.array([
    (-1, -1),
    (1, -1),
    (1, 1),
    (-1, 1)], dtype=np.float)
cell = np.array([(0, 1, 2, 3)], dtype=np.int)
quadtree = Quadtree(point, cell)
mesh = PolygonMesh(point, cell)
integrator = QuadrangleQuadrature(6)
vem = PoissonVEMModel(model, mesh, p=p, integrator=integrator, quadtree=quadtree)
