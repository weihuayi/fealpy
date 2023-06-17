import numpy as np
import matplotlib.pyplot as plt
import time

from fealpy.pde.darcy_forchheimer_2d import PolyData
#from fealpy.pde.darcy_forchheimer_2d import ExponentData
from fealpy.pde.darcy_forchheimer_2d import SinsinData
#from fealpy.pde.darcy_forchheimer_2d_1 import CoscosData1
from fealpy.fdm.DarcyForchheimerFDMModel import DarcyForchheimerFDMModel
#from fealpy.fdm.DarcyForchheimerFDMModel_pu import DarcyForchheimerFDMModel_pu
#from fealpy.fdm.DarcyForchheimerFDMModelpu import DarcyForchheimerFDMModel
#from fealpy.fdm.testDarcyForchheimerFDMModelpu import DarcyForchheimerFDMModel
from fealpy.tools.show import showmultirate
from fealpy.tools.showsolution import showsolution

box = [0,1,0,1]
mu = 2
k = 1
rho = 1
beta = 30
tol = 1e-9
nx = 4
ny = 4
pde = PolyData(box,mu,k,rho,beta,tol)
#pde = ExponentData(box)
#pde = SinsinData(box,mu,k,rho,beta,tol)
maxit = 4
Ndof = np.zeros((maxit,), dtype=np.int)
errorType = ['$|| u_I - u_h||_0$','$||p_I - p_h||_0$',\
        '$||Dp_I -  Dp_h||_0$','$||Dp1_I - Dp1_h||_0$','$|| |u_I| - |u_h|||$']
errpL2 = np.zeros((maxit,), dtype=np.float)
err = np.zeros((2,maxit),dtype=np.float)
error = np.zeros((5,maxit),dtype=np.float)
count = np.zeros((maxit,), dtype=np.int)
for i in range(maxit):
    t1 = time.time()
    mesh = pde.init_mesh(nx,ny)
    fdm = DarcyForchheimerFDMModel(pde,mesh)
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    Ndof[i] = NE + NC
    t2 = time.time()
    count[i],r= fdm.solve()
    Stime = time.time() - t2
    ue,pe = fdm.get_max_error()
    err[0,i] = ue
    err[1,i] = pe

    ueL2,peL2 = fdm.get_L2_error()
    normuL2 = fdm.get_normu_error()
    DpeL2 = fdm.get_DpL2_error()
    Dp1eL2 = fdm.get_Dp1L2_error()

    error[0,i] = ueL2
    error[1,i] = peL2
    error[2,i] = DpeL2
    error[3,i] = Dp1eL2
    error[4,i] = normuL2
#    showsolution(plt, mesh, pde, uh, ph)
    x = np.arange(count[i])
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_title('rp')
    ax1.scatter(x,r[0,:count[i]],c='r',marker = '.')
    ax2 = fig.add_subplot(122)
    ax2.set_title('ru')
    ax2.scatter(x,r[1,:count[i]],c='b',marker = '.')

    if i < maxit - 1:
        nx = 2*nx
        ny = 2*ny
    tottime = time.time() -t1
    print('Total time:',tottime)
    print('Solve time:',Stime)
print('err',err)
print('error',error)
print('iter',count)
#mesh.add_plot(plt,cellcolor='w')
#showmultirate(plt,0,Ndof,err,errorType)
showmultirate(plt,0,Ndof,error,errorType)
plt.show()
