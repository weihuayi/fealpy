import numpy as np
from simp_compliance import TopSimp

ts = TopSimp()
mesh = ts.mesh
NC = mesh.number_of_cells()
rho = np.ones((NC)) - 0.5
print("rho:", rho.shape, "\n", rho)
E = ts.materialModel(rho)
print("E:", E.shape, "\n", E)
space = ts.space
gdof = space.number_of_global_dofs()
print("gdof:", gdof)

uh = ts.fe(rho)
print("uh:", uh.shape, "\n", uh)
