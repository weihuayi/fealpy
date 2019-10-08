import numpy as np

from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.mesh.simple_mesh_generator import triangle
from 


pde = CosCosData()
domain = pde.domain()
h = 0.1
mesh = triangle(domain, h, meshtype='polygon')

space = ConformingVirtualElementSpace2d(mesh, p=1, q=4)

M = space.mass_matrix()

