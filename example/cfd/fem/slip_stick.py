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
rho_gas = 1
rho_melt = 1
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
    x_minmax_bounds = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]#[[1,4],[7,10]]#[[1,3],[4,6],[7,9]]#
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
    val = x-0.2
    return val

def changerho(fun, s0):
    tag_m = s0 <=0
    tag_g = s0 > 0
    fun[tag_m] = rho_melt
    fun[tag_g] = rho_gas
    return fun

def changemu(fun, s0):
    tag_m = s0<=0
    tag_g = s0 > 0
    fun[tag_m] = mu_melt
    fun[tag_g] = mu_gas
    return fun

def level_x(phi, y):
    ipoint = phi.space.interpolation_points()
    y_indices = np.where(ipoint[:, 1]==y)[0]
    phi_y = phi[y_indices]
    sort_indeces = np.argsort(np.abs(phi_y))[:2]
    indices = y_indices[sort_indeces]
    if phi[indices[0]] < 1e-8:
        return ipoint[indices[0],0]
    else :
        zong = np.abs(phi[indices[0]]) + np.abs(phi[indices[1]])
        ws0 = 1 - np.abs(phi[indices[0]])/zong
        ws1 = 1 - np.abs(phi[indices[1]])/zong
        val = ws0 * ipoint[indices[0], 0] + ws1*ipoint[indices[1],0]
        return val

uspace = LagrangeFESpace(mesh,p=udegree, doforder='sdofs')
pspace = LagrangeFESpace(mesh,p=pdegree, doforder='sdofs')
u0 = uspace.function(dim = udim)
u1 = uspace.function(dim = udim)

p0 = pspace.function()
p1 = pspace.function()

phi0 = uspace.interpolate(dist)

mu = uspace.function()
rho = uspace.function()
mu = changemu(mu,phi0)
rho = changerho(rho,phi0)

NSSolver = NSFEMSolver(mesh, dt,uspace,pspace,mu,rho)
LSSolver = LSFEMSolver(uspace)
stress = NSSolver.netwon_sigma(u0, mu)
name = ['velocity','mu','phi','rho','D00','D11','D01','minus']
variable = [u0,mu,phi0,rho,stress[0,0,:],
        stress[1,1,:],stress[0,1,:],stress[0,0,:]-stress[1,1,:]]
NSSolver.output(name,variable,0)

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
is_ux_bdof[is_uslip_bdof] = False

ipoint = uspace.interpolation_points()[is_uin_bdof]
uinflow = u_inflow_dirichlet(ipoint)
xx[0:ugdof][is_uin_bdof] = uinflow[:, 0]
xx[ugdof:2*ugdof][is_uin_bdof] = uinflow[:,1]
isBdDof = np.hstack([is_ux_bdof, is_u_bdof, is_p_bdof])

for i in range(0, nt):
    #下一个时间层
    t1 = tmesh.next_time_level()
    print("t1=", t1)

    A = NSSolver.ossen_A(u0)
    b = NSSolver.ossen_b(u0)


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
    
    xxx = level_x(phi0, 0)
    print("边界点位置:",xxx)

    mu = changemu(mu,phi0)
    rho = changerho(rho,phi0)
    
    if i%step == 0:
        stress = NSSolver.netwon_sigma(u0, mu)
        phi0 = LSSolver.reinit(phi0=phi0, dt=0.0001,eps=2e-4, nt=100, alpha=alpha)
        name = ['velocity','mu','phi','rho','D00','D11','D01','minus']
        variable = [u0,mu,phi0,rho,stress[0,0,:],
                stress[1,1,:],stress[0,1,:],stress[0,0,:]-stress[1,1,:]]
        NSSolver.output(name,variable,i)
    
    u0[:] = u1 
    p0[:] = p1
    # 时间步进一层 
    tmesh.advance()
