import numpy as np
import matplotlib.pyplot as plt
from fealpy.pde.poisson_2d import CosCosData
from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.functionspace import MonomialSpace2d

pde = CosCosData()
quadtree = pde.init_mesh(4, meshtype='quadtree')
p = 3


for i in range(4):
    pmesh = quadtree.to_pmesh()
    space = ConformingVirtualElementSpace2d(pmesh, p=p)
    uh = space.interpolation(pde.solution)
    sh = space.project_to_smspace(uh)
#    mspace = MonomialSpace2d(pmesh, p=p)
#    vh = mspace.projection(sh)
#    sh = space.smspace.projection(vh)
    error = space.integralalg.L2_error(pde.solution, sh.value)
    print(error)
    quadtree.uniform_refine()
