import numpy as np
from simp_compliance_test import TopSimp

ts = TopSimp()
mesh = ts.mesh
NC = mesh.number_of_cells()
rho = np.ones((NC)) - 0.5
print("rho:", rho.shape, "\n", rho)
E = ts.material_model(rho)
print("E:", E.shape, "\n", E)
space = ts.space
gdof = space.number_of_global_dofs()
print("gdof:", gdof)

uh, _ = ts.fe_analysis(rho)
print("uh:", uh.shape, "\n", uh)

c1, c2 = ts.compute_compliance(rho)
print("c1:", c1, "c2:", c2)

dc = ts.compute_compliance_sensitivity(rho)
print("dc:", dc)