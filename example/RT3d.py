import numpy as np


from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import ScaledMonomialSpace3d 


mesh = MF.one_tetrahedron_mesh()

space = ScaledMonomialSpace3d(mesh, p=1)

space.show_cell_basis_index(p=4)
