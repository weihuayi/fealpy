import sys
import numpy as np  
import matplotlib.pyplot as plt
import scipy.io as sio

from fealpy.fem.PoissonFEMModel import PoissonFEMModel

from fealpy.tools.show import showmultirate, show_error_table

d = int(sys.argv[1])

if d == 1:
    from fealpy.pde.poisson_1d import CosData as PDE 
elif d==2:
    from fealpy.pde.poisson_2d import CosCosData as PDE 
elif d==3:
    from fealpy.pde.poisson_3d import CosCosCosData as PDE 


p = int(sys.argv[2])
n = int(sys.argv[3])
maxit = int(sys.argv[4])

pde = PDE()
errorType = ['$|| u - u_h||_0$', '$||\\nabla u - \\nabla u_h||_0$']

errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)

Ndof = np.zeros(maxit, dtype=np.int)

mesh = pde.init_mesh(n)
integrator = mesh.integrator(p+2)

for i in range(maxit):
    fem = PoissonFEMModel(pde, mesh, p, integrator)
    ls = fem.solve()
    sio.savemat('test%d.mat'%(i), ls)
    Ndof[i] = fem.femspace.number_of_global_dofs()
    errorMatrix[0, i] = fem.get_L2_error()
    errorMatrix[1, i] = fem.get_H1_error()
    if i < maxit - 1:
        mesh.uniform_refine()

show_error_table(Ndof, errorType, errorMatrix)
showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()
