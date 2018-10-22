import numpy as np
import matplotlib.pyplot as plt
import time

from fealpy.pde.darcy_forchheimer_2d import CoscosData1
#from fealpy.fdm.DarcyForchheimerFDMModel import DarcyForchheimerFDMModel
#from fealpy.fdm.DarcyForchheimerFDMModelpu import DarcyForchheimerFDMModel
from fealpy.fdm.testDarcyForchheimerFDMModelpu import DarcyForchheimerFDMModel
from fealpy.tools.show import showmultirate
from fealpy.tools.showsolution import showsolution

box = [0,1,0,1]
<<<<<<< HEAD
<<<<<<< HEAD
nx = 64
ny = 64
||||||| merged common ancestors
nx = 128
ny = 128
=======
||||||| merged common ancestors
=======
<<<<<<< HEAD
nx = 4
ny = 4
||||||| merged common ancestors
nx = 128
ny = 128
=======
>>>>>>> e459bb9b45daa737e75766d4a6f7803e2e9905f9
nx = 256 
ny = 256 
<<<<<<< HEAD
>>>>>>> 14a86c65b99a6b9c9cbd0ef98e4bbe936af644e6
||||||| merged common ancestors
=======
>>>>>>> 69274f3204aeb7f5e3adbf00770529b605236071
>>>>>>> e459bb9b45daa737e75766d4a6f7803e2e9905f9
pde = CoscosData1(box)
maxit = 4
Ndof = np.zeros((maxit,), dtype=np.int)
errorType = ['$|| u_I - u_h||_0$','$||p_I - p_h||_0$','$||Dp_I - Dp_h||_0$']
errpL2 = np.zeros((maxit,), dtype=np.float)
err = np.zeros((2,maxit),dtype=np.float)
error = np.zeros((2,maxit),dtype=np.float)
count = np.zeros((maxit,), dtype=np.int)
for i in range(maxit):
    t = time.time()
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
#    DpeL2 = fdm.get_DpL2_error()
 #   print('I',I)
#    print('ep',ep)
#    print('psemi',psemi)
    error[0,i] = ueL2
    error[1,i] = peL2
#    error[2,i] = DpeL2
#    showsolution(plt, mesh, pde, uh, ph)
    if i < maxit - 1:
        nx = 2*nx
        ny = 2*ny
    elapsed = time.time() -t
    print(elapsed)
print('err',err)
print('error',error)
print('iter',count)
#mesh.add_plot(plt,cellcolor='w')
#showmultirate(plt,0,Ndof,err,errorType)
showmultirate(plt,0,Ndof,error,errorType)
plt.show()
