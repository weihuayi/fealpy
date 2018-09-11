import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.darcy_2d import CoscosData
from fealpy.fdm.DarcyFDMModel import DarcyFDMModel
from fealpy.tools.show import showrate

box = [0,1,0,1]
nx = 0.25
ny = 0.25
pde = CoscosData(box)
mesh = pde.init_mesh(nx,ny)
maxit = 4
Ndof = np.zeros((maxit,), dtype=np.int)
errorType = ['$|| u - u_h||_0$']
erruL2 = np.zeros((maxit,), dtype=np.float)
errpL2 = np.zeros((matit,), dtype=np.float)
for i in range(maxit):
    fdm = DarcyFDMModel(pde,mesh)
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    
showrate(axes,0,4,error,option)
plt.show()
