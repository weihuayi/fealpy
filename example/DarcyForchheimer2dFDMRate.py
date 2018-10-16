import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.darcy_forchheimer_2d import CoscosData1
<<<<<<< HEAD
from fealpy.fdm.DarcyForchheimerFDMModel import DarcyForchheimerFDMModel
#from fealpy.fdm.DarcyForchheimerFDMModelpu import DarcyForchheimerFDMModel
#from fealpy.fdm.DarcyForchheimerFDMModel_pu import DarcyForchheimerFDMModel
||||||| merged common ancestors
#from fealpy.fdm.DarcyForchheimerFDMModel import DarcyForchheimerFDMModel
from fealpy.fdm.DarcyForchheimerFDMModelpu import DarcyForchheimerFDMModel
=======
from fealpy.fdm.DarcyForchheimerFDMModel import DarcyForchheimerFDMModel
#from fealpy.fdm.DarcyForchheimerFDMModelpu import DarcyForchheimerFDMModel
>>>>>>> 99da71a9a1166f73488a5da276ebe59f8e088465
from fealpy.tools.show import showmultirate
from fealpy.tools.showsolution import showsolution

box = [0,1,0,1]
<<<<<<< HEAD
nx = 128
ny = 128
||||||| merged common ancestors
nx = 4
ny = 4
=======
nx = 32
ny = 32
>>>>>>> 99da71a9a1166f73488a5da276ebe59f8e088465
pde = CoscosData1(box)
<<<<<<< HEAD
maxit = 3
||||||| merged common ancestors
maxit = 1
=======
maxit = 4
>>>>>>> 99da71a9a1166f73488a5da276ebe59f8e088465
Ndof = np.zeros((maxit,), dtype=np.int)
errorType = ['$|| u_I - u_h||_0$','$||p_I - p_h||_0$','$||Dp_I - Dp_h||_0$']
erruL2 = np.zeros((maxit,), dtype=np.float)
errpL2 = np.zeros((maxit,), dtype=np.float)
err = np.zeros((2,maxit),dtype=np.float)
error = np.zeros((3,maxit),dtype=np.float)
count = np.zeros((maxit,), dtype=np.int)
for i in range(maxit):
    mesh = pde.init_mesh(nx,ny)
    fdm = DarcyForchheimerFDMModel(pde,mesh)
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    Ndof[i] = NE + NC
    count[i]= fdm.solve()
    ue,pe = fdm.get_max_error()
    err[0,i] = ue
    err[1,i] = pe
    ueL2,peL2 = fdm.get_L2_error()
<<<<<<< HEAD
    I = fdm.get_DpL2_error()
    print('I',I)
#    print('ep',ep)
#    print('psemi',psemi)
||||||| merged common ancestors
=======
    DpeL2 = fdm.get_DpL2_error()
>>>>>>> 99da71a9a1166f73488a5da276ebe59f8e088465
    error[0,i] = ueL2
    error[1,i] = peL2
<<<<<<< HEAD
    error[2,i] = I
||||||| merged common ancestors
=======
    error[2,i] = DpeL2
    print('NE',NE)
>>>>>>> 99da71a9a1166f73488a5da276ebe59f8e088465
#    showsolution(plt, mesh, pde, uh, ph)
    if i < maxit - 1:
        nx = 2*nx
        ny = 2*ny
print('err',err)
print('error',error)
print('iter',count)
#mesh.add_plot(plt,cellcolor='w')
#showmultirate(plt,0,Ndof,err,errorType)
showmultirate(plt,0,Ndof,error,errorType)
plt.show()
