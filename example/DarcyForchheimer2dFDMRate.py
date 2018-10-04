import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.darcy_forchheimer_2d import CoscosData1
from fealpy.pde.darcy_2d_1 import CoscosData
from fealpy.fdm.DarcyForchheimerFDMModel import DarcyForchheimerFDMModel
from fealpy.tools.show import showmultirate

box = [0,1,0,1]
nx = 4
ny = 4
pde = CoscosData1(box)
pde1 = CoscosData(box)
maxit = 4
Ndof = np.zeros((maxit,), dtype=np.int)
errorType = ['$|| u_I - u_h||_0$','$p_I - p_h$']
erruL2 = np.zeros((maxit,), dtype=np.float)
errpL2 = np.zeros((maxit,), dtype=np.float)
err = np.zeros((2,maxit),dtype=np.float)
error = np.zeros((2,maxit),dtype=np.float)
count = np.zeros((maxit,), dtype=np.int)
for i in range(maxit):
    mesh = pde.init_mesh(nx,ny)
    fdm = DarcyForchheimerFDMModel(pde1,pde,mesh)
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    Ndof[i] = NE + NC
    fdm.solve()
    count[i] = fdm.cycling()
    ue,pe = fdm.get_max_error()
    err[0,i] = ue
    err[1,i] = pe
    ueL2,peL2 = fdm.get_L2_error()
    error[0,i] = ueL2
    error[1,i] = peL2
    if i < maxit - 1:
        nx = 2*nx
        ny = 2*ny
print('err',err)
print('error',error)
print('iter',count)
mesh.add_plot(plt,cellcolor='w')
showmultirate(plt,0,Ndof,err,errorType)
showmultirate(plt,0,Ndof,error,errorType)
plt.show()
