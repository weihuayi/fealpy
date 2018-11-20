import numpy as np
import matplotlib.pyplot as plt
from fealpy.mg.DarcyForchheimerP0P1 import DarcyForchheimerP0P1
from fealpy.mesh.adaptive_tools import mark
from fealpy.tools.show import showmultirate, show_error_table
from fealpy.pde.darcy_forchheimer_2d import LShapeRSinData
from fealpy.pde.darcy_forchheimer_2d import DarcyForchheimerdata1
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

box = [-1,1,-1,1]
mu = 1
rho = 1
beta = 10
alpha = 1/beta
tol = 1e-6
maxN = 2000
p = 1
n = 2

pde = LShapeRSinData (mu, rho, beta, alpha, tol, maxN)
mesh = pde.init_mesh(n)
#integrator1 = mesh.integrator(p+2)
#integrator0 = mesh.integrator(p+1)
#
#theta = 0.35
#errorType = ['$|| u - u_h||_0$','$|| p - p_h||$', '$||\\nabla p - \\nabla p_h||_0$']
#
#maxit = 4
#errorMatrix = np.zeros((len(errorType), maxit), dtype=np.float)
#Ndof = np.zeros(maxit,dtype = np.int)
#
#
#
#fem = DarcyForchheimerP0P1(pde, mesh, integrator0, integrator1)
#fem.solve()
#
#fig = plt.figure()
#fig.set_facecolor('white')
#axes = fig.gca(projection='3d')
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
x, y = np.meshgrid(x, y)
pi = np.pi
theta = np.arctan2(y, x)
theta = (theta >= 0)*theta + (theta < 0)*(theta+2*pi)
p = (x*x + y*y)**(1/3)*np.sin(2/3*theta)


fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, p, rstride=1, cstride=1, cmap=cm.viridis)
   
    
    
    
    
#    NC = mesh.number_of_cells()
#    NN = mesh.number_of_edges()
#    Ndof[i] = 2*NC+NN
#    errorMatrix[0, i] = fem.get_uL2_error()
#    errorMatrix[1, i] = fem.get_pL2_error()
#    errorMatrix[2, i] = fem.get_H1_error()
#    eta1
#    eta2
#    markedCell = mark(eta, theta=theta)
#    if i < maxit -1:
#        markedCell = mark(eta, theta=theta)
#        mesh.bisect(markedCell)
#mesh.add_plot(plt, cellcolor='w')
#
#showmultirate(plt, 0, Ndof, errorMatrix, errorType)
plt.show()

