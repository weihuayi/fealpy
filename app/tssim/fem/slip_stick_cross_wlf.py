#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: slip_stick.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2023年12月29日 星期五 13时00分56秒
	@bref 
	@ref 
'''  
import numpy as np
from scipy.sparse.linalg import spsolve
from fealpy.mesh import TriangleMesh
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.functionspace import LagrangeFESpace
from fealpy.decorator import barycentric, cartesian
from fealpy.cfd import NSFEMSolver 
from fealpy.levelset import LSFEMSolver

#时空参数
udegree = 2
pdegree = 1
ns = 10
nt = 3000
T = 30
output = './'
step = 5

# 模型参数
alpha = 0.625/ns
'''
rho_melt = 1020
#rho_gas = 0.001*rho_melt
rho_gas = rho_melt
mu_gas = 1.792e-5
#mu_melt = 4.7e-5
mu_melt = mu_gas
'''
rho_gas = 1
rho_melt = 1
#mu_gas = 1.792e-5
mu_gas = 0.01
mu_melt = 0.01

# 定义网格、空间
udim = 2
h = 1
domain = [0, 10*h, 0, h]
mesh = TriangleMesh.from_box([domain[0],domain[1],domain[2],domain[3]], 10*ns, ns)
tmesh = UniformTimeLine(0,T,nt)
dt = tmesh.dt
eps = 1e-12

#定义初边界条件
@cartesian
def is_outflow_boundary(p):
    return np.abs(p[..., 0]-domain[1])<eps

@cartesian
def is_inflow_boundary(p):
    return np.abs(p[..., 0]-domain[0])<eps

@cartesian
def is_wall_boundary(p):
    return (np.abs(p[..., 1]-domain[3])<eps)|\
           (np.abs(p[..., 1]-domain[4])<eps)

@cartesian
def is_slip_boundary(p):    
    x = p[..., 0]
    y = p[..., 1]
    x_minmax_bounds = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    x_minmax_bounds2 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    mn, mx = np.array(x_minmax_bounds).T
    mn2 ,mx2 = np.array(x_minmax_bounds2).T
    val = x[:, None]
    result1 = ((val > mn) & (val < mx) ).any(1)& (y-domain[2]<eps)
    result2 = ((val > mn2) & (val < mx2)).any(1) & (np.abs(y-domain[3])<eps)
    result3 = ((y-domain[2]) < eps)|((y-domain[3]) < eps)
    return ((result1) | (result2)) & (result3)

@cartesian
def u_inflow_dirichlet(p):
    x = p[..., 0]
    y = p[..., 1]
    value = np.zeros(p.shape)
    value[..., 0] = 2*y*(h-y)
    value[..., 1] = 0
    return value

def dist(p):
    x = p[...,0]
    y = p[...,1]
    val = x-0.1
    return val

def changerho(fun, s0):
    tag_m = s0 <=0
    tag_g = s0 > 0
    fun[tag_m] = rho_melt
    fun[tag_g] = rho_gas
    return fun

def changemu(mufun, s0):
    tag_m = s0 <=0
    tag_g = s0 >  0
    mufun[tag_g] = mu_gas
    mufun[tag_m] = mu_melt
    return mufun

uspace = LagrangeFESpace(mesh,p=udegree, doforder='sdofs')
phi0 = uspace.interpolate(dist)

## 初始网格
## 加密
for i in range(5):
    cell2dof = mesh.cell_to_ipoint(udegree)
    phi0c2f = phi0[cell2dof]
    isMark = np.abs(np.mean(phi0c2f,axis=-1))< 0.05
    data = {'phi0':phi0c2f} 
    option = mesh.bisect_options(data=data)
    mesh.bisect(isMark,options=option)

    uspace = LagrangeFESpace(mesh,p=udegree,doforder='sdofs')
    cell2dof = uspace.cell_to_dof()
    phi0 = uspace.function()
    phi0[cell2dof.reshape(-1)] = option['data']['phi0'].reshape(-1)

phi0 = uspace.interpolate(dist)
pspace = LagrangeFESpace(mesh,p=pdegree, doforder='sdofs')

u0 = uspace.function(dim = udim)
u1 = uspace.function(dim = udim)

p0 = pspace.function()
p1 = pspace.function()

mu = uspace.function()
rho = uspace.function()
rho = changerho(rho,phi0)

NSSolver = NSFEMSolver(mesh, dt,uspace,pspace,rho)
LSSolver = LSFEMSolver(uspace)
@barycentric
def cross_mu(bcs, index=None):
    return NSSolver.cross_wlf(p0, u0, bcs, T=200)

#mu = changemu(mu,phi0)
mu[uspace.cell_to_dof()] = uspace.interpolate(cross_mu)


stress = NSSolver.netwon_sigma(u0, mu)
name = ['velocity','mu','phi','rho','D00','D11','D01','minus']
variable = [u0,mu,phi0,rho,stress[0,0,:],
        stress[1,1,:],stress[0,1,:],stress[0,0,:]-stress[1,1,:]]
NSSolver.output(name,variable,0)


for i in range(0, nt):
    #下一个时间层
    t1 = tmesh.next_time_level()
    print("t1=", t1)
 
    ##边界条件
    ugdof = uspace.number_of_global_dofs()
    gdof = 2*uspace.number_of_global_dofs() + pspace.number_of_global_dofs()
    xx = np.zeros(gdof, np.float64)

    is_u_bdof = uspace.is_boundary_dof()
    is_ux_bdof = uspace.is_boundary_dof()
    is_uslip_bdof = uspace.is_boundary_dof(threshold = is_slip_boundary)
    is_uin_bdof = uspace.is_boundary_dof(threshold = is_inflow_boundary)
    is_uout_bdof = uspace.is_boundary_dof(threshold = is_outflow_boundary)
    is_p_bdof = pspace.is_boundary_dof(threshold = is_outflow_boundary)

    is_u_bdof[is_uout_bdof] = False
    is_ux_bdof[is_uout_bdof] = False
    #is_ux_bdof[is_uslip_bdof] = False

    ipoint = uspace.interpolation_points()[is_uin_bdof]
    uinflow = u_inflow_dirichlet(ipoint)
    xx[0:ugdof][is_uin_bdof] = uinflow[:, 0]
    xx[ugdof:2*ugdof][is_uin_bdof] = uinflow[:,1]
    isBdDof = np.hstack([is_ux_bdof, is_u_bdof, is_p_bdof])
    
    @barycentric
    def cross_mu(bcs, index=None):
        return NSSolver.cross_wlf(p0, u0, bcs, T=200)
     
    A = NSSolver.ossen_A(u0, mu, rho)
    b = NSSolver.ossen_b(u0, rho)

    ## 边界条件处理
    b -= A@xx
    b[isBdDof] = xx[isBdDof]
    #ctx.set_centralized_sparse(A)
    x = b.copy()
    #ctx.set_rhs(x)
    x[~isBdDof] = spsolve(A[:,~isBdDof][~isBdDof,:],x[~isBdDof])
    #ctx.run(job=6)

    u1[0, :] = x[0:ugdof]
    u1[1, :] = x[ugdof:2*ugdof]
    p1[:] = x[2*ugdof:]
    
    #levelset
    phi0[:] = LSSolver.mumps_solve(4, phi0, dt, u1)
    
    ucell2dof = uspace.cell_to_dof()
    pcell2dof = pspace.cell_to_dof()
    for j in range(5): 
        phi0c2f = phi0[ucell2dof]
        u1xc2f = u1[0,:][ucell2dof]
        u1yc2f = u1[1,:][ucell2dof]
        p1c2f = p1[pcell2dof]
        cellmeasure = mesh.entity_measure('cell')
        isMark = np.abs(np.mean(phi0c2f,axis=-1))<0.05
        isMark = np.logical_and(np.mean(phi0c2f,axis=-1)>-0.01,isMark)
        isMark = np.logical_and(isMark,cellmeasure>4e-5)
        data = {'phi0':phi0c2f,'u1x':u1xc2f,'u1y':u1yc2f,'p1':p1c2f} 
        option = mesh.bisect_options(data=data,disp=False)
        mesh.bisect(isMark,options=option)

        uspace = LagrangeFESpace(mesh,p=udegree, doforder='sdofs')
        pspace = LagrangeFESpace(mesh,p=pdegree, doforder='sdofs')
        ucell2dof = uspace.cell_to_dof()
        pcell2dof = pspace.cell_to_dof()
        phi0 = uspace.function()
        u1 = uspace.function(dim=2)
        p1 = pspace.function()
        phi0[ucell2dof.reshape(-1)] = option['data']['phi0'].reshape(-1)
        u1[0,:][ucell2dof.reshape(-1)] = option['data']['u1x'].reshape(-1)
        u1[1,:][ucell2dof.reshape(-1)] = option['data']['u1y'].reshape(-1)
        p1[pcell2dof.reshape(-1)] = option['data']['p1'].reshape(-1)
    
    
    #重新粗化
    for j in range(5):
        phi0c2f = phi0[ucell2dof]
        u1xc2f = u1[0,:][ucell2dof]
        u1yc2f = u1[1,:][ucell2dof]
        p1c2f = p1[pcell2dof]
        cellmeasure = mesh.entity_measure('cell')
        isMark = np.abs(np.mean(phi0c2f,axis=-1))>0.05
        isMark = np.logical_and(np.mean(phi0c2f,axis=-1)<0.01,isMark)
        data = {'phi0':phi0c2f,'u1x':u1xc2f,'u1y':u1yc2f,'p1':p1c2f} 
        option = mesh.bisect_options(data=data,disp=False)
        mesh.coarsen(isMark,options=option)

        uspace = LagrangeFESpace(mesh,p=udegree, doforder='sdofs')
        pspace = LagrangeFESpace(mesh,p=pdegree, doforder='sdofs')
        ucell2dof = uspace.cell_to_dof()
        pcell2dof = pspace.cell_to_dof()
        phi0 = uspace.function()
        u1 = uspace.function(dim=2)
        p1 = pspace.function()
        phi0[ucell2dof.reshape(-1)] = option['data']['phi0'].reshape(-1)
        u1[0,:][ucell2dof.reshape(-1)] = option['data']['u1x'].reshape(-1)
        u1[1,:][ucell2dof.reshape(-1)] = option['data']['u1y'].reshape(-1)
        p1[pcell2dof.reshape(-1)] = option['data']['p1'].reshape(-1)
 
    u0 = uspace.function(dim = udim)
    p0 = pspace.function()
    u0[:] = u1 
    p0[:] = p1
    
    NSSolver = NSFEMSolver(mesh, dt, uspace, pspace)
    LSSolver = LSFEMSolver(uspace)
    mu = uspace.function()
    rho = uspace.function()
    rho = changerho(rho,phi0)
    #mu = changemu(mu, phi0)
    
    mu[uspace.cell_to_dof()] = uspace.interpolate(cross_mu)
    if i%step == 0:
        stress = NSSolver.netwon_sigma(u1, mu)
        phi0 = LSSolver.reinit(phi0=phi0, dt=0.0001,eps=2e-4, nt=100, alpha=alpha)
        name = ['velocity','mu','phi','rho','D00','D11','D01','minus']
        variable = [u0,mu,phi0,rho,stress[0,0,:],
                stress[1,1,:],stress[0,1,:],stress[0,0,:]-stress[1,1,:]]
        NSSolver.output(name,variable,i)
    
    # 时间步进一层 
    tmesh.advance()
