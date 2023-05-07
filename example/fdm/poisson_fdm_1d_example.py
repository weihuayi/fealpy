import numpy as np
from scipy.sparse.linalg import spsolve
# import ipdb
import matplotlib.pyplot as plt
from fealpy.pde.elliptic_1d import SinPDEData
from fealpy.mesh import UniformMesh1d


nx = 5 
maxit = 5
#ipdb.set_trace()
pde = SinPDEData()
domain = pde.domain()
em = np.zeros((3, maxit), dtype=np.float64)
fig, axes = plt.subplots()
for i in range(maxit):
    nx *= 2
    hx = (domain[1] - domain[0])/nx
    mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])
    uh = mesh.function('node') 
    f = mesh.interpolate(pde.source, intertype='node')
    A = mesh.laplace_operator() 
    A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f, uh=uh)

    uh[:] = spsolve(A, f)
    mesh.show_function(axes, uh)

    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)

print(em[:, 0:-1]/em[:, 1:])

#fig, axes = plt.subplots()
#mesh.add_plot(axes)
#mesh.find_node(axes, showindex=True)
plt.show()
