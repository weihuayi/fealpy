import numpy as np
from simp_compliance import TopSimp

ts = TopSimp()
mesh = ts.mesh
NC = mesh.number_of_cells()
print("cell_area:", mesh.entity_measure('cell'))
rho = np.ones((NC)) - 0.5 
print("rho:", rho.shape, "\n", rho)
E = ts.material_model(rho)
print("E:", E.shape, "\n", E)
space = ts.space
gdof = space.number_of_global_dofs()
print("gdof:", gdof)

uh = ts.fe_analysis(rho)
print("uh:", uh.shape, "\n", uh)
