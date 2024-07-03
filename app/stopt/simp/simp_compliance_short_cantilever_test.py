import numpy as np
from simp_compliance_test import TopSimp

ts = TopSimp()
mesh = ts.mesh
NC = mesh.number_of_cells()
rho = np.full((NC, ), ts.global_volume_constraints['volfrac'])
print("rho:", rho.shape, "\n", rho)
E = ts.material_model(rho)
print("E:", E.shape, "\n", E)
space = ts.space
gdof = space.number_of_global_dofs()
print("gdof:", gdof)
print("fixeddofs:", ts.bc['fixeddofs'])

uh, ue = ts.fe_analysis(rho)
print("uh:", uh.shape, "\n", uh.round(4))
print("ue:", ue.shape, "\n", ue.round(4))

c1, c2 = ts.compute_compliance(rho)
print("c1:", c1, "c2:", c2)

# dc = ts.compute_compliance_sensitivity(rho)
# print("dc:", dc)