import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
#import h5py

from fealpy.pde.darcy_forchheimer_2d import PolyData
from fealpy.pde.darcy_forchheimer_2d import ExponentData
from fealpy.pde.darcy_forchheimer_2d import SinsinData
from fealpy.pde.darcy_forchheimer_2d import ArctanData
from fealpy.fdm.NonDFFDMModel import NonDFFDMModel
from fealpy.fdm.NonDFFDMModel_pu import NonDFFDMModel_pu
from fealpy.fdm.NonDFFDMModel_normu import NonDFFDMModel_normu
from fealpy.tools.show import showmultirate
box = [0,1,0,1]
mu = 2
k = 1
rho = 1
beta = 10
tol = 1e-9
hx = np.array([0.12,0.34,0.22,0.32])
hy = np.array([0.25,0.13,0.33,0.29])
#hy = np.array([0.16,0.23,0.32,0.11,0.18])
#hx = np.array([0.12,0.34,0.45,0.09])
#hy = np.array([0.25,0.13,0.54,0.08])
#hy = np.array([0.25,0.13,0.34,0.20,0.08])
#hx = np.array([0.25,0.25,0.25,0.25])
#hy = np.array([0.25,0.25,0.25,0.25])
#hy = np.array([0.2,0.2,0.2,0.2,0.2])
#m = 16
#hx = hx/m
#hy = hy/m
#hx = hx.repeat(m)
#hy = hy.repeat(m)

pde = PolyData(box,mu,k,rho,beta,tol)
#pde = ExponentData(box,mu,k,rho,beta,tol)
#pde = SinsinData(box,mu,k,rho,beta,tol)
#pde = ArctanData(box,mu,k,rho,beta,tol)
maxit = 4
Ndof = np.zeros((maxit,), dtype=np.int)
errorType1 = ['$|| u - u_h||_0$','$||p - p_h||_0$',\
        '$||Dp -  Dp_h||_0$','$||Dp1 - Dp1_h||_0$']
errorType2 = ['$|| (|u| - |Qu)u|||_0$','$|||u|u - |Qu|u_h||_0$',\
        '$|| |u| - |Qu|||_0$']#,'$|||u|u - |Qu|u||_0/||u||_0$']

err = np.zeros((2,maxit),dtype=np.float)
error1 = np.zeros((4,maxit),dtype=np.float)
error2 = np.zeros((3,maxit),dtype=np.float)
count = np.zeros((maxit,), dtype=np.int)
for i in range(maxit):
    t1 = time.time()
    mesh = pde.init_mesh(hx,hy)
    fdm = NonDFFDMModel(pde,mesh)
#    fdm = NonDFFDMModel_pu(pde,mesh)
#    fdm = NonDFFDMModel_normu(pde,mesh)
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    Ndof[i] = NE + NC
    t2 = time.time()
    count[i],r,u,p= fdm.solve()
    x = np.r_[u, p]
    data = {"up":x}
    sio.savemat('data.mat', data)
    Stime = time.time() - t2
    ue,pe = fdm.get_max_error()
    err[0,i] = ue
    err[1,i] = pe
    ueL2,peL2 = fdm.get_L2_error()
    DpeL2 = fdm.get_DpL2_error()
    Dp1eL2 = fdm.get_Dp1L2_error()
    uqunorm = fdm.get_uqunorm_error()
    uuqunorm = fdm.get_uuqunorm_error()
    uqnorm = fdm.get_uqnorm_error()
 #   print('I',I)
#    print('ep',ep)
#    print('psemi',psemi)
    error1[0,i] = ueL2
    error1[1,i] = peL2
    error1[2,i] = DpeL2
    error1[3,i] = Dp1eL2
    error2[0,i] = uqunorm
    error2[1,i] = uuqunorm
    error2[2,i] = uqnorm

#    x = np.arange(count[i])
#    fig = plt.figure()
#    ax1 = fig.add_subplot(121)
#    f,(ax,ax3) = plt.subplots(2,1,sharex=True)
#
#    ax.plot(r[0,count[i]-1])
#    ax3.plot(r[0,count[i]-1])
#    ax.set_ylim(0.0007,0.0008)
#    ax3.set_ylim(0,0.00000001)
#
##hide the spines between ax and ax2
#    ax.spines['bottom'].set_visible(False)
#    ax3.spines['top'].set_visible(False)
#    ax.xaxis.tick_top()
#    ax.tick_params(labeltop=False)
#    ax3.xaxis.tick_bottom()
#
#    d = .015
#    kwargs = dict(transform=ax.transAxes,color='k', clip_on=False)
#    ax.plot((-d, +d), (-d,+d),**kwargs)
#    ax.plot((1-d,1+d), (-d,+d), **kwargs)
#
#    kwargs.update(transform=ax3.transAxes)
#    ax3.plot((-d, +d), (1-d,1+d),**kwargs)
#    ax3.plot((1-d,1+d), (-d,+d), **kwargs)
#    
#    ax1.set_title('rp')
#    ax1.scatter(x,r[0,:count[i]],c='r',marker = '.')
#    plt.text(count[i]/2, r[0,count[i]-1]+0.00002,'rp = %e' %r[0,count[i]-1],ha = 'center')
#    plt.yscale('symlog')
#    ax1.set_yticks([-0.001,0,0.0002, 0.0008, 0.0009])
#    plt.ylim([-0.0001,0.0009])
#    ax2 = fig.add_subplot(122)
#    ax2.set_title('ru')
#    ax2.scatter(x,r[1,:count[i]],c='b',marker = '.')

    if i < maxit - 1:
        hx = hx/2
        hx = hx.repeat(2)
        hy = hy/2
        hy = hy.repeat(2)
    tottime = time.time() -t1
    print('Total time:',tottime)
    print('Solve time:',Stime)
print('err',err)
print('error1',error1)
print('error2',error2)
print('iter',count)
print('Ndof',Ndof)
print('M4P4b30t9')
#mesh.add_plot(plt,cellcolor='w')
#showmultirate(plt,0,Ndof,err,errorType)
showmultirate(plt,0,Ndof,error1,errorType1)
#plt.savefig("/home/liao/Desktop/M4P4b30t9updp.eps",dpi=None)
showmultirate(plt,0,Ndof,error2,errorType2)
#plt.savefig("/home/liao/Desktop/M4P4b30t9ucu.eps",dpi=None)
plt.show()

