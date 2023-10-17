#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: navier_stokes_wlf.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年09月01日 星期五 09时33分08秒
	@bref 
	@ref 
'''  
import numpy as np
from  mumps import DMumpsContext
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, bmat
from mumps import DMumpsContext

from fealpy.decorator import cartesian,barycentric
from fealpy.mesh import TriangleMesh
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import LagrangeFiniteElementSpace

from fealpy.fem import ScalarDiffusionIntegrator, VectorMassIntegrator
from fealpy.fem import ScalarMassIntegrator, ScalarConvectionIntegrator
from fealpy.fem import VectorViscousWorkIntegrator, PressWorkIntegrator
from fealpy.fem import BilinearForm, MixedBilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import VectorSourceIntegrator, ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from scipy.sparse.linalg import spsolve
from fealpy.mesh.vtk_extent import write_to_vtu
from scipy.sparse import csr_matrix

from tssim.part import *

T = 40
nt = 1000

# 网格,空间,函数
node = np.array([
    [0.0,0.0],
    [2.5,0.0],
    [2.5,0.75],
    [6.0,0.75],
    [6.0,1.0],
    [2.5,1.0],
    [0.0,1.0],
    [0.0,0.75]],dtype = np.float64)

cell = np.array([
    [1,2,0],
    [7,0,2],
    [3,4,2],
    [5,2,4],
    [6,7,5],
    [2,5,7]],dtype = np.int_)

mesh = TriangleMesh(node,cell)
mesh.uniform_refine(4)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

tmesh = UniformTimeLine(0,T,nt)
dt = tmesh.dt
eps = 1e-10
maxit = 3  #非线性迭代次数
step = 5
output = './'

rho = 1000 
Cp = 0.05
kappa = 1

@cartesian
def is_outflow_boundary(p):
    return np.abs(p[..., 0] - 6) < eps

@cartesian
def is_inflow_boundary(p):
    return np.abs(p[..., 0] - 0) < eps

@cartesian
def is_up_boundary(p):
    return np.abs(p[..., 1] - 1) < eps


@cartesian
def is_wall_boundary(p):
    x = p[...,0] 
    y = p[...,1]
    result0 = np.abs(y - 0.0) < eps
    mn, mx = np.array([[0.0,0.75]]).T
    val = y[:, None]
    result1 = ((val > mn) & (val < mx)).any(1)
    result1 = (np.abs(x - 2.5) < eps) & (result1)
    mn, mx = np.array([[2.5,6.0]]).T
    val = x[:, None]
    result2 = ((val > mn) & (val < mx)).any(1)
    result2 = (np.abs(y-0.75)<eps) & (result2)
    #result3 =  np.abs(p[..., 1] - 1) < eps
    return (result0) | (result1) | (result2) 

@cartesian
def u_inflow_dirichlet(p):
    x = p[...,0]
    y = p[...,1]
    value = np.zeros(p.shape)
    value[...,0] = 0.03*y*(2-y)
    value[...,1] = 0
    return value
################
def power(gamma,T,gamma0=1e-4,beta=0.01,T_ref=50,m=1e4,n=0.5):
    tag = gamma<=gamma0
    mu = np.zeros(gamma.shape)
    mu[tag] = m*np.power(gamma0,n-1)*np.exp(-beta*(T[tag]-T_ref))
    mu[~tag] = m*np.power(gamma[~tag],n-1)*np.exp(-beta*(T[~tag]-T_ref))
    return mu

def de(u,bcs):
    deformnation = u.grad_value(bcs)
    deformnation = 1/2*(deformnation + deformnation.transpose(0,1,3,2))
    return deformnation

def ga(d):
    val = np.sqrt(2*np.einsum('ijkl,ijkl->ij',d,d))
    return val

def haosan(u,bcs):
    gradu = u.grad_value(bcs)
    deformnation = gradu + gradu.transpose(0,1,3,2)
    val = np.einsum('ijkl,ijkl->ij',deformnation,gradu)
    return val
##############

space2 = LagrangeFESpace(mesh, p=2, doforder='sdofs')
space1 = LagrangeFESpace(mesh, p=1)

## 初始网格
u0 = space2.function(dim=2)
us = space2.function(dim=2)
u1 = space2.function(dim=2)
p0 = space1.function()
p1 = space1.function()
T0 = space2.function()
T0[:] = 50
T1 = space2.function()

#####
pdegree = 1
udegree = 2
tdegree = 2
udim = 2
pspace = LagrangeFiniteElementSpace(mesh,p=pdegree)
uspace = LagrangeFiniteElementSpace(mesh,p=udegree)
tspace = LagrangeFiniteElementSpace(mesh,p=tdegree)

nu0 = uspace.function(dim=udim)
nus = uspace.function(dim=udim)
nu1 = uspace.function(dim=udim)

np0 = pspace.function()
np1 = pspace.function()

nT0 = tspace.function()
nT0[:] = 50
nTs = tspace.function()
nT1 = tspace.function()

qf = mesh.integrator(5)
bcs,ws = qf.get_quadrature_points_and_weights()
assemble = Assemble(mesh, 5)
pgdof = pspace.number_of_global_dofs()
ugdof = uspace.number_of_global_dofs()
tgdof = tspace.number_of_global_dofs()
####


mesh.nodedata['velocity'] = u0.T
mesh.nodedata['pressure'] = p0
mesh.nodedata['tempture'] = T0

fname = output + 'test_0000000000.vtu'
mesh.to_vtk(fname=fname)
ctx = DMumpsContext()
ctx.set_silent()

for i in range(1):
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)
     
    deformnation = de(nu0,bcs)
    gamma = ga(deformnation)
    mu = power(gamma,nT0(bcs))
    phi_h = haosan(nu0,bcs) 
    
    gdof2 = space2.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()
    gdof = 2*gdof2 + gdof1 + gdof2
    # 动量方程矩阵
    BF0 = BilinearForm(space2)        
    
    BF0.add_domain_integrator(ScalarMassIntegrator(1/dt*rho)) 
    
    @barycentric
    def etafun(bcs, index):
        # POWER-LAW
        deformnation = u0.grad_value(bcs, index)
        deformnation = 1/2*(deformnation + deformnation.transpose(0,2,1,3))
        gamma = np.sqrt(2*np.einsum('ijkl,ikjl->il',deformnation,deformnation))
        tag = gamma <= 1e-4
        result = np.zeros_like(gamma)
        result[tag] = 1e4 * 1e-4**(-0.5) *np.exp(-0.01*(T0(bcs, index)[tag]-50)) 
        result[~tag] = 1e4 *np.power(gamma[~tag],-0.5)*np.exp(-0.01*(T0(bcs, index)[~tag]-50)) 
        return  result

    #BF0.add_domain_integrator(ScalarDiffusionIntegrator(etafun)) 
    AU = BF0.assembly()
    
    # 连续方程矩阵 
    BF1 = MixedBilinearForm((space1,), 2*(space2,)) 
    BF1.add_domain_integrator(PressWorkIntegrator()) 
    AP = BF1.assembly()
    
    BF2 = BilinearForm(space1)
    BF2.add_domain_integrator(ScalarDiffusionIntegrator())
    ASP = BF2.assembly()  
    
    # 能量方程矩阵
    BF3 = BilinearForm(space2)
    
    @barycentric 
    def dt_rho_C(bcs,index):
        return rho*Cp/dt
    BF3.add_domain_integrator(ScalarMassIntegrator(dt_rho_C))
    BF3.add_domain_integrator(ScalarDiffusionIntegrator(kappa)) 
    
    AT = BF3.assembly() 
    ########################

    M = assemble.matrix([udegree, 0], [udegree, 0], rho)
    Sx = assemble.matrix([pdegree,0],[udegree,1])
    Sy = assemble.matrix([pdegree,0],[udegree,2])

    MT = assemble.matrix([tdegree, 0], [tdegree, 0], rho*Cp)
    ST0 = assemble.matrix([tdegree,1], [tdegree,1], kappa) 
    ST1 = assemble.matrix([tdegree,2], [tdegree,2], kappa)
    ST = ST0+ST1
    ######################

    # 右端向量
    LFU = LinearForm(2*(space2,))
    
    @barycentric 
    def dt_rho_u0(bcs, index):
        return rho*u0(bcs,index)/dt
    
    LFU.add_domain_integrator(VectorSourceIntegrator(dt_rho_u0))
    bu = LFU.assembly()
    bp = np.zeros(space1.number_of_global_dofs())
    ## 能量方程右端向量 
    LFT = LinearForm(space2) 
    
    @barycentric
    def bT_source(bcs, index=np.s_[:]):
        gu = u0.grad_value(bcs, index)
        deformnation = gu + gu.transpose(0,2,1,3)
        result = np.einsum('ijkl,ijkl->il',deformnation, gu)
        return etafun(bcs,index)*result
    LFT.add_domain_integrator(ScalarSourceIntegrator(bT_source)) 
    
    @barycentric 
    def dt_rho_C_T0(bcs,index):
        return rho*Cp*T0(bcs,index)/dt 
    LFT.add_domain_integrator(ScalarSourceIntegrator(dt_rho_C_T0))
    bT = LFT.assembly() 
    b = np.hstack([bu,bp,bT])
    ############## 
    
    S1 = assemble.matrix([udegree,1], [udegree,1], mu)
    S2 = assemble.matrix([udegree,2], [udegree,2], mu)
    S = S1+S2
    
    #组装右端 
    nb1 = 1/dt*np.hstack((M@nu0[:,0],M@nu0[:,1])).T
    nb2 = np.zeros(pgdof)
    nb3 = 1/dt*MT@T0+assemble.vector([tdegree, 0],mu*phi_h)
    nb = np.hstack([nb1,nb2,nb3])
    #############3
    
    # 边界处理
    is_ux_bdof = space2.is_boundary_dof()
    is_uy_bdof = space2.is_boundary_dof()
    is_uin_bdof = space2.is_boundary_dof(threshold = is_inflow_boundary)
    is_uout_bdof = space2.is_boundary_dof(threshold = is_outflow_boundary)
    is_uup_bdof = space2.is_boundary_dof(threshold = is_up_boundary)

    is_pout_bdof = space1.is_boundary_dof(threshold = is_outflow_boundary)

    is_t_bdof = space2.is_boundary_dof()
    is_tin_bdof = space2.is_boundary_dof(threshold = is_inflow_boundary)
    is_tout_bdof = space2.is_boundary_dof(threshold = is_outflow_boundary)
    is_twall_bdof = space2.is_boundary_dof(threshold = is_wall_boundary)
    is_tup_bdof = space2.is_boundary_dof(threshold = is_up_boundary)

    is_ux_bdof[is_uup_bdof] = False 
    is_ux_bdof[is_uout_bdof] = False 
    #is_uy_bdof[is_uup_bdof] = False 

    is_p_bdof = is_pout_bdof
    is_t_bdof[is_tout_bdof] = False
    is_t_bdof[is_tup_bdof] = False
    xx = np.zeros(gdof, np.float64)

    ipoint = space2.interpolation_points()[is_uin_bdof]
    uinflow = u_inflow_dirichlet(ipoint)
    xx[0:gdof2][is_uin_bdof] = uinflow[:,0]
    xx[gdof2:2*gdof2][is_uin_bdof] = uinflow[:,1]
    xx[-gdof2:][is_tin_bdof] = 50
    xx[-gdof2:][is_twall_bdof] = 100

    isBdDof = np.hstack([is_ux_bdof,is_uy_bdof,is_p_bdof,is_t_bdof])
    bdIdx = np.zeros(gdof, dtype=np.int_)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    
    us[:] = u0
    for zz in range(maxit):
        BF0C = BilinearForm(space2)

        @barycentric 
        def rho_u(bcs,index):
            return rho*us(bcs,index)
        BF0C.add_domain_integrator(ScalarConvectionIntegrator(rho_u))
        AUC = BF0C.assembly()
 
        BF3C = BilinearForm(space2)
        @barycentric 
        def rho_C_u(bcs,index):
            return rho*Cp*us(bcs,index)
        BF3C.add_domain_integrator(ScalarConvectionIntegrator(c=rho_C_u))
        ATU = BF3C.assembly() 
        
        
        A0 = bmat([[AU+S+AUC, None],[None, AU+S+AUC]], format='csr')  
        
        A = bmat([[A0,  -AP, None],\
                [AP.T, 1e-8*ASP, None],\
                [None,None,AT+ATU]], format='csr')
        AA = bmat([[A0,  -AP],\
                [AP.T, 1e-8*ASP]],format='csr')

        A = T@A + Tbd
        b[isBdDof] = xx[isBdDof]
        ctx.set_centralized_sparse(A)
        x = b.copy()
        ctx.set_rhs(x)
        ctx.run(job=6)
         
        #################
        D0 = assemble.matrix([udegree,1], [udegree,0], rho*nus(bcs)[...,0])
        D1 = assemble.matrix([udegree,2], [udegree,0], rho*nus(bcs)[...,1])
         
        grad_Ts = nTs.grad_value(bcs)
       
        DT0 = assemble.matrix([tdegree,1], [tdegree, 0], nus(bcs)[...,0]*rho*Cp) 
        DT1 = assemble.matrix([tdegree,2], [tdegree, 0], nus(bcs)[...,1]*rho*Cp) 
        nA = bmat([[1/dt*M + S+D0+D1, None, -Sx, None],\
                [None, 1/dt*M + S +D0+D1, -Sy, None],\
                [Sx.T, Sy.T, 1e-8*ASP, None],\
                [None, None, None, 1/dt*MT+ST+DT0+DT1]], format='csr')
        
        
        nAA = bmat([[1/dt*M + S+D0+D1, None, -Sx],\
                [None, 1/dt*M + S +D0+D1, -Sy],\
                [Sx.T, Sy.T, 1e-8*ASP]],format='csr')
        nAAA = np.vstack((Sx.toarray(),Sy.toarray()))
        print(np.sum(np.abs(nAAA-AP)))
        #print(np.sum(np.abs(nAA-AA))) 

        #边界条件处理及solver    
        nA = T@nA + Tbd
        nb[isBdDof] = xx[isBdDof]
        ctx.set_centralized_sparse(nA)
        x = nb.copy()
        ctx.set_rhs(x)
        ctx.run(job=6)
        
        nu1[:,0] = x[0:ugdof]
        nu1[:,1] = x[ugdof:2*ugdof]
        np1[:] = x[2*ugdof:-tgdof]
        nT1[:] = x[-tgdof:]
        nus[:] = nu1 
        nTs[:] = nT1
        #################
        u1[0,:] = x[0:gdof2]
        u1[1,:] = x[gdof2:2*gdof2]
        p1[:] = x[2*gdof2:-gdof2]
        T1[:] = x[-gdof2:]
    
        us[:] = u1 
    
    u0[:] = u1 
    p0[:] = p1
    T0[:] = T1
    
    nu0[:] = nu1 
    np0[:] = np1
    nT0[:] = nT1
    #print("压力：",np.abs(np.sum(np1))) 
    
    if i%step == 0:
        fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
        mesh.nodedata['velocity'] = u1.T
        mesh.nodedata['pressure'] = p1
        mesh.nodedata['tempture'] = T1
        mesh.to_vtk(fname=fname)
    # 时间步进一层 
    tmesh.advance()
ctx.destroy()

'''
测试levelset
@cartesian
def velocity_field(p):
    x = p[..., 0]
    y = p[..., 1]
    u = np.zeros(p.shape)
    u[..., 0] = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    u[..., 1] = -np.sin((np.pi*y))**2 * np.sin(2*np.pi*x)
    return u

@cartesian
def circle(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    return val

mesh = TriangleMesh.from_unit_square(nx=50, ny=50)
timeline = UniformTimeLine(0, 1, 100)
dt = timeline.dt

space = LagrangeFESpace(mesh, p=2)
phi0 = space.interpolate(circle)
u = space.interpolate(velocity_field, dim=2)


for i in range(100):
        
    t1 = timeline.next_time_level()
    print("t1=", t1)
    
    phi0[:] = level_set(dt = dt, phi0=phi0 ,u=u)
    
    fname = './' + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata = {'phi':phi0, 'velocity':u}
    mesh.to_vtk(fname=fname)
    if i%30 == 0  and i!=0: 
       rein(phi0, 1/50)

    # 时间步进一层 
    timeline.advance()
'''
