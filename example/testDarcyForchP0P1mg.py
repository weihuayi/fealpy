import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.sparse.linalg import cg, inv, spsolve
from scipy.sparse import eye, spdiags
from fealpy.mg.DarcyForchheimerP0P1 import DarcyForchheimerP0P1
from fealpy.mg.DFP0P1mg import DFP0P1mg
from fealpy.mg.DarcyP0P1 import DarcyP0P1
from fealpy.mg.DarcyForchP0P1mg import DarcyForchP0P1mg
from fealpy.tools.show import showmultirate, show_error_table
from fealpy.pde.darcy_forchheimer_2d import DarcyForchheimerdata1

box = [-1,1,-1,1]
mu = 1
rho = 1
beta = 10
alpha = 1/beta
tol = 1e-6
level = 3
mg_maxN = 3
J = level -2
maxN = 2000
p = 1
n = 2

pde = DarcyForchheimerdata1(box,mu,rho,beta,alpha,level,tol,maxN,mg_maxN,J)
## Generate an initial mesh
mesh = pde.init_mesh(n+level)
NC = mesh.number_of_cells()

integrator1 = mesh.integrator(p+2)
integrator0 = mesh.integrator(p+1)

## Initial guess
#NC = mesh.number_of_cells()
mfem = DarcyP0P1(pde, mesh, 1, integrator1)
u,p = mfem.solve()
#A = mfem.get_left_matrix()
#A11 = A[:2*NC,:2*NC]
#A12 = A[:2*NC,2*NC:]
#A21 = A[2*NC:,:2*NC]
#b = mfem.get_right_vector()
#cellmeasure = mesh.entity_measure('cell')
#area = np.tile(cellmeasure,2)
#Aalpha = A11 + spdiags(area/alpha, 0, 2*NC, 2*NC)
#Aalphainv = spdiags(1/Aalpha.data, 0, 2*NC, 2*NC)
#Ap = A21@Aalphainv@A12

## MG iteration
m = 0;
error = np.ones(maxN, dtype=np.float)
residual = np.ones(maxN, dtype=np.float)
Ndof = np.zeros(maxN,dtype = np.int)

while residual[m] > tol and m < maxN:
    
    uold = np.zeros(u.shape)
    uold[:] = u
#    femg = DarcyForchP0P1mg(n, pde, integrator0, integrator1)
    femg = DFP0P1mg(n, pde, integrator0, integrator1)
    u,p,rn = femg.solve()
    m = m + 1
    erru = norm(u - uold)/norm(uold)
    error[m] = erru
    residual[n] = rn
erruIuh = femg.get_uIL2_error()
errp = femg.get_uL2_error()
errpH1 = femg.get_H1_error()
plot(np.arange(n),residual[:n],c='r',maker='.')


plt.show()
