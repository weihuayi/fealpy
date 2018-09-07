import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.darcy_2d import CoscosData
from fealpy.fdm.DarcyFDMModel import DarcyFDMModel
from fealpy.tools.show import showrate

n = 4
box = [0,1,0,1]
nx = 0.25
ny = 0.25
pde = CoscosData(box,nx,ny)
mesh = pde.init_mesh(n=n,meshtype='quad')
maxit = 4
Ndof = np.zeros((maxit,), dtype=np.int)
errorType = ['$|| u - u_h||_0$']
errorMatrix = np.zeros((maxit,2), dtype=np.float)
for i in range(maxit):
    fdm = DarcyFDMModel(pde,mesh)
    print(fdm)
    fdm.solve()
    for j in range(2):
        errorMatrix[i,j] = fdm.get_L2_error()
print(errorMatrix)
showrate(axes,0,4,error,option)
plt.show()
