#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年03月04日 星期六 14时19分10秒
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.mesh import MeshFactory as MF
from fealpy.decorator import cartesian,barycentric
from scipy.sparse.linalg import spsolve
from mumps import DMumpsContext
from scipy.sparse import spdiags

lag = 1.0
TT = 0.0005
eps1 = 0.001
eps2 = 0.01

volTarg = 0.5
lagrangesetp = 8

plotNum = int(10)
N = int(400)
NiterLSE = (10)
ReInitFreq = (5)
Ninit = int(10)


mesh = MF.boxmesh2d([0, 1, 0, 1], nx=80, ny=80, meshtype='tri')
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
#plt.show()

def Phi0(p):
    x = p[..., 0]
    y = p[..., 1]
    val = 0.2-(np.abs(x-0.5)-0.5)*(np.abs(y-0.5)-0.5)
    return val

space =LagrangeFiniteElementSpace(mesh, p=1)

phi = space.interpolation(Phi0)
uold = space.function()

h = max(mesh.entity_measure('edge'))
print("最大网格尺寸:", h)

centerbcs = np.array([[0.3333,0.3333,0.3333]])
cell2dof = space.cell_to_dof()

S = space.stiff_matrix()
M = space.mass_matrix()

gdof = space.number_of_global_dofs()
isBdDof = space.is_boundary_dof()
bdIdx = np.zeros(gdof, dtype=np.int_)
bdIdx[isBdDof] = 1
Tbd = spdiags(bdIdx, 0, gdof, gdof)
T = spdiags(1-bdIdx, 0, gdof, gdof)

qf = mesh.integrator(3)
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')

ctx = DMumpsContext()
ctx.set_silent()


for i in range(1): #gradient_N
    print("interation = ",i)
    
    nabla = space.function()
    S = sapce.function()
    M1 = space.function()
    M2 = space.function()
    phiinit = space.function()
    
    if i%ReInitFreq == 0:
        gradphi = phi.grad_value(centerbcs)
        gradphi_x = space.function()
        gradphi_y = space.function()
        
        np.add.at(gradphi_x,cell2dof.reshape(-1),gradphi[0,:,0].repeat(3))
        np.add.at(gradphi_y,cell2dof.reshape(-1),gradphi[0,:,1].repeat(3))
        
        num = np.zeros_like(gradphi_x, dtype = int)
        np.add.at(num, cell2dof.reshape(-1) ,1)
        gradphi_x[:] /= num
        gradphi_y[:] /= num
        
        nabla[:] = gradphi_x[:]**2 + gradphi_y[:]**2
        S[:] = phi[:]/np.sqrt(phi[:]**2 + h*h*nabla[:])
        M1 = gradphi_x[:]/np.sqre(nabla[:] + eps1**2)
        M2 = gradphi_y[:]/np.sqre(nabla[:] + eps1**2)
        
        #phiinit = covect();
        phiinit += TT*S[:]
        phi[:] = phiinit[:]

    # 计算gradphi_x,gradphi_y
    gradphi = phi.grad_value(centerbcs)
    gradphi_x = space.function()
    gradphi_y = space.function()
    
    np.add.at(gradphi_x,cell2dof.reshape(-1),gradphi[0,:,0].repeat(3))
    np.add.at(gradphi_y,cell2dof.reshape(-1),gradphi[0,:,1].repeat(3))
    
    num = np.zeros_like(gradphi_x, dtype = int)
    np.add.at(num, cell2dof.reshape(-1) ,1)
    gradphi_x[:] /= num
    gradphi_y[:] /= num
    
    #nabla,N1,N2
    nabla[:] = gradphi_x[:]**2 + gradphi_y[:]**2

    N1 = space.function()
    N2 = space.function()

    N1[:] = gradphi_x[:]/np.sqrt(nabla[:] + eps1**2)
    N2[:] = gradphi_y[:]/np.sqrt(nabla[:] + eps1**2)

    X = space.function()
    Xtag = phi<0
    X[Xtag] = 1

    Dirac = space.function()
    Dtag = np.abs(phi) < 0.01
    Dirac[Dtag] = 1
    
    #解状态方程
    A0 = S
    b0 = space.source_vector(X)    
    
    A0 = T@A0 + Tbd 
    b0[isBdDof] = 0
    ctx.set_centralized_sparse(A0)
    x0 = b0.copy()
    ctx.set_rhs(x0)
    ctx.run(job=6)

    u = space.function()
    u[:] = x0

    #计算目标函数
    compliance = -0.5*np.einsum('i, ik, ik, j-> ', ws, X(bcs), u(bcs), cellmeasure)
    objective = compliance

    #计算体积误差
    volume = np.einsum('i, ik, j-> ', ws, X(bcs), cellmeasure)
    volErr = np.abs(volume - volTarg)
    print("体积误差:", volErr)

    #解速度长光滑化方程
    V = space.function()
    V[:] = -X[:] * -u[:] - lag*X[:]

    A1 = eps2*S + M
    b1 = space.source_vector(V)

    ctx.set_centralized_sparse(A1)
    x1 = b1.copy()
    ctx.set_rhs(x1)
    ctx.run(job=6)

    Vreg = space.function()
    Vreg[:] = x1
    vv = space.function()
    vv[:] = np.abs(Vreg[:])
    Vmax = np.max(vv[:])
    print("......", Vmax)
    
    V[:] = Vreg[:]
    T = 0.02*h/np.max(V[:])
    if(T < 0.00001):
        T = 0.0001

    Vn1 = space.function()
    Vn2 = space.function()
    Vn1 = V[:]*N1[:]
    Vn2 = V[:]*N2[:]

    print("下降步长 T=", T)
    '''
    for j in range(NiterLSE):
        phi = convect();
    '''

    perimeter = np.einsum('i, ik, ik, j-> ', ws, Dirac(bcs), nabla(bcs)**0.5, cellmeasure)
    temp = np.einsum('i, ik, j-> ', ws, -Dirac(bcs)*nabla(bcs)**0.5*u(bcs),cellmeasure)
    lag = 0.5*lag - 0.5*temp/perimeter + lagrangesetp*(volume - volTarg)/volTarg 
ctx.destroy()
