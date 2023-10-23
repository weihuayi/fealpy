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

def level_set(dt, phi0, u,  M=None): 
    space = phi0.space
    BF = BilinearForm(space)
    BF.add_domain_integrator(ScalarConvectionIntegrator(c=u))
    if M is None:
        BF2 = BilinearForm(space)
        BF2.add_domain_integrator(ScalarMassIntegrator())
        M = BF2.assembly() 
        C = BF.assembly() 
        A = M + dt/2*C
    else:
        C = BF.assembly()
        A = M + dt/2*C
    
    b = M@phi0 - dt/2*(C@phi0)
    phi0[:] = spsolve(A, b)
    return phi0


def rein(phi0, cellscale, dt=0.0001, eps=1e-4, nt=4, alpha=None):
    ###
    # paulo Level Set Implementation Using FEniCS
    ###
    if alpha is None:
        alpha = 0.0625*cellscale
    space = phi0.space

    phi1 = space.function()
    phi2 = space.function()
    phi1[:] = phi0
    
    BF = BilinearForm(space)
    BF.add_domain_integrator(ScalarDiffusionIntegrator())
    S = BF.assembly() 
    
    BF2 = BilinearForm(space)
    BF2.add_domain_integrator(ScalarMassIntegrator())
    M = BF2.assembly() 
    eold = 0   

    for i in range(nt):
        @barycentric
        def f(bcs, index):
            grad = phi1.grad_value(bcs)
            val = 1 - np.sqrt(np.sum(grad**2, -1))
            #sign = phi0(bcs)/np.sqrt(phi0(bcs)**2+cellscale**2)
            #val *= sign
            val *= np.sign(phi0(bcs))
            return val
        
        LF = LinearForm(space)
        LF.add_domain_integrator(ScalarSourceIntegrator(f))
        b0 = LF.assembly()
        b = M@phi1 + dt*b0 - dt*alpha*(S@phi1)

        phi2[:] = spsolve(M, b)
        error = space.mesh.error(phi2, phi1)
        print("重置:", error) 
        if eold < error and error< eps :
            break
        else:
            phi1[:] = phi2
            eold = error
    return phi1



def heaviside(phi, epsilon, lvalue, rvalue):
    '''
    epsilon 界面厚度
    phi 距离函数
    '''
    space = phi.space
    fun = space.function()
    tag = (-epsilon<= phi)  & (phi <= epsilon)
    tag1 = phi > epsilon
    fun[tag1] = 1
    fun[tag] = 0.5*(1+phi[tag]/epsilon) 
    fun[tag] += 0.5*np.sin(np.pi*phi[tag]/epsilon)/np.pi
    fun[:] = rvalue + (rvalue-lvalue)*fun
    return fun 


T = 10
nt = 2000
ns = 10
h = 1
domain = [0,10*h,0,h]
mesh = TriangleMesh.from_box(box=domain, nx=10*ns, ny=ns)
dx = 1/ns
tmesh = UniformTimeLine(0,T,nt)
dt = tmesh.dt
eps = 1e-10
epsilon = dx  #界面一半的厚度 
maxit = 3  #非线性迭代次数
step = 5
output = './'
udegree = 2
pdegree = 1
tdegree = 2

wrho = 1020
grho = wrho*0.001
wC = 1700
gC = 0.588*wC
wlambda = 0.173
glambda = 0.139*wlambda
geta = 1.792e-5
weta = geta/0.001
We = 8995
Pe = 2505780.347
Re = 0.01
Br = 147398.844

@cartesian
def is_outflow_boundary(p):
    return np.abs(p[..., 0] - domain[1]) < eps

@cartesian
def is_inflow_boundary(p):
    return np.abs(p[..., 0] - domain[0]) < eps

@cartesian
def is_wall_boundary(p):
    return (np.abs(p[..., 1] - domain[3]) < eps) | \
           (np.abs(p[..., 1] - domain[2]) < eps)

@cartesian
def is_stick_boundary(p):
    y = p[...,1]
    x = p[...,0]
    #x_minmax_bounds = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0],[6.0,7.0],[8.0,9.0]]
    x_minmax_bounds = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],[7.0,8.0],[9.0,10.0]]
    mn, mx = np.array(x_minmax_bounds).T
    val = x[:, None]
    result = ((val > mn) & (val < mx)).any(1)
    result2 = ((y - domain[3]) < eps) | ((y- domain[2]) < eps)
    return (result) & (result2)

@cartesian
def dist(p):
    x = p[...,0]
    y = p[...,1]
    #val =  (x-h) - 2*h*y*(h-y)/h**2
    val =  x-0.1
    return val
           
@cartesian
def u_inflow_dirichlet(p):
    x = p[...,0]
    y = p[...,1]
    value = np.zeros(p.shape)
    value[...,0] = 5
    value[...,1] = 0
    return value

def level_x(phi,y):
    ns = min(phi.space.mesh.entity_measure('edge'))
    near = np.abs(phi) < np.sqrt(2*ns**2) 
    point = phi.space.interpolation_points()
    a = point[near] #界面附近的点
    x = point[near][a[:,1]==y][:,0]
    return np.mean(x)


space2 = LagrangeFiniteElementSpace(mesh, p=2)
space1 = LagrangeFiniteElementSpace(mesh, p=1)

## 加密
phi0 = space2.interpolation(dist)
for i in range(5):
    cell2dof = mesh.cell_to_ipoint(2)
    phi0c2f = phi0[cell2dof]
    isMark = np.abs(np.mean(phi0c2f,axis=-1))< epsilon
    data = {'phi0':phi0c2f} 
    option = mesh.bisect_options(data=data, disp=False)
    mesh.bisect(isMark,options=option)

    space2 = LagrangeFiniteElementSpace(mesh, p=2)
    space1 = LagrangeFiniteElementSpace(mesh, p=1)
    cell2dof = space2.cell_to_dof()
    phi0 = space2.function()
    phi0[cell2dof.reshape(-1)] = option['data']['phi0'].reshape(-1)

u0 = space2.function(dim=2)
us = space2.function(dim=2)
u1 = space2.function(dim=2)
p0 = space1.function()
p1 = space1.function()
T0 = space2.function()
Ts = space2.function()
T1 = space2.function()
phi0 = space2.interpolation(dist)


## 初始网格
ctx = DMumpsContext()
ctx.set_silent()

for i in range(nt):
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)
    
    rhofun =  heaviside(phi0, epsilon, wrho, grho)
    Cfun =  heaviside(phi0, epsilon, wC, gC)
    lambdafun =  heaviside(phi0, epsilon, wlambda, glambda)
    etafun =  heaviside(phi0, epsilon, weta, geta)

    if i%1 == 0:
        fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
        mesh.nodedata['velocity'] = u1
        mesh.nodedata['pressure'] = p1
        mesh.nodedata['tempture'] = T1
        mesh.nodedata['rho'] = rhofun
        mesh.nodedata['surface'] = phi0
        mesh.nodedata['比热容'] = Cfun
        mesh.nodedata['热扩散系数'] = lambdafun
        mesh.to_vtk(fname=fname)
    
    qf = mesh.integrator(5)
    bcs,ws = qf.get_quadrature_points_and_weights()
    assemble = Assemble(mesh, 5)
     
    gdof2 = space2.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()
    gdof = 2*gdof2 + gdof1 + gdof2
    us[:] = u0
    Ts[:] = T0
    
    M = assemble.matrix([udegree, 0], [udegree, 0], rhofun(bcs))
    Sx = assemble.matrix([pdegree,0],[udegree,1])
    Sy = assemble.matrix([pdegree,0],[udegree,2])
 
    MT = assemble.matrix([tdegree, 0], [tdegree, 0], rhofun(bcs)*Cfun(bcs))
    #MT = Pe*assemble.matrix([tdegree, 0], [tdegree, 0], rhofun(bcs)*Cfun(bcs))
    ST0 = assemble.matrix([tdegree,1], [tdegree,1], lambdafun(bcs)) 
    ST1 = assemble.matrix([tdegree,2], [tdegree,2], lambdafun(bcs))
    ST = ST0+ST1
    
    SP = space1.stiff_matrix()
    
    @barycentric
    def etafun2(bcs):
        # crosswlf
        eta0 = 1.9e11*np.exp(-27.396*(T0(bcs )-417.15)/(51.6+(T0(bcs )-417.15)))
        deformnation = u0.grad_value(bcs )
        deformnation = 1/2*(deformnation + deformnation.transpose(0,1,3,2))
        gamma = np.sqrt(2*np.einsum('ijkl,ijlk->ij',deformnation,deformnation))
        result = eta0/(1+(eta0*gamma/182680)**(1-0.574))
        #heaviside
        pbcs = phi0(bcs) 
        tag = (-epsilon<= pbcs)  & (pbcs <= epsilon)
        tag1 = pbcs > epsilon
        result[tag1] = geta
        result[tag] = 0.5*(1+pbcs[tag]/epsilon) 
        result[tag] += 0.5*np.sin(np.pi*pbcs[tag]/epsilon)/np.pi
        return  result
    S1 = assemble.matrix([udegree,1], [udegree,1], etafun(bcs))
    S2 = assemble.matrix([udegree,2], [udegree,2], etafun(bcs))
    #S = (S1+S2)/Re
    S = S1+S2
    
    #组装右端 
    bu = 1/dt*np.hstack((M@u0[:,0],M@u0[:,1])).T
    bp = np.zeros(gdof1)
    
    @barycentric
    def bT_source(bcs):
        gradu = u0.grad_value(bcs)
        D = 1/2*(gradu + gradu.transpose(0,1,3,2))
        etaD = np.einsum('ij,ijkl->ijkl',etafun(bcs), D)
        etaD[:,:,0,0] -= p0(bcs)
        etaD[:,:,1,1] -= p0(bcs)
        val1 = np.einsum('ijkc,ijkc->ij',etaD, gradu)
        return val1
    bT = assemble.vector([tdegree, 0],bT_source(bcs))
    bT += 1/dt*(MT@T0)
    
    b = np.hstack([bu,bp,bT])
    
         
    # 边界处理
    is_ux_bdof = space2.is_boundary_dof()
    is_uy_bdof = space2.is_boundary_dof()
    is_p_bdof = space1.is_boundary_dof()
    is_T_bdof = space2.is_boundary_dof()
    xux = np.zeros_like(is_ux_bdof, np.float64)
    xuy = np.zeros_like(is_uy_bdof, np.float64)
    xp = np.zeros_like(is_p_bdof, np.float64)
    xT = np.zeros_like(is_T_bdof, np.float64)

    ## 速度边界处理  
    is_uout_bdof = space2.is_boundary_dof(threshold = is_outflow_boundary)
    is_uin_bdof = space2.is_boundary_dof(threshold = is_inflow_boundary)
    
    is_ux_bdof[is_uout_bdof] = False 
    is_uy_bdof[is_uout_bdof] = False 
    
    is_ux_bdof[is_uin_bdof] = False 
    is_uy_bdof[is_uin_bdof] = False 
    #xux[is_uin_bdof] = 5
    ## 压力边界条件处理
    @cartesian
    def is_melt_boundary(p):
        levelx = level_x(phi0, 0)    
        return np.abs(p[..., 0] - levelx) < eps    
    is_pout_bdof = space1.is_boundary_dof(threshold = is_outflow_boundary)
    is_pin_bdof = space1.is_boundary_dof(threshold = is_inflow_boundary)
    is_pwall_bdof = space1.is_boundary_dof(threshold = is_wall_boundary)
    is_pmelt_bdof = space1.is_boundary_dof(threshold = is_melt_boundary) 
    is_pmelt_bdof = is_pmelt_bdof & is_pwall_bdof        
    
    #is_p_bdof[is_pmelt_bdof] = False 
    is_p_bdof[is_pwall_bdof] = False 
    xp[is_pin_bdof] = 8
    xp[is_pout_bdof] = 0

    ## 温度边界条件处理   
    is_Tout_bdof = space2.is_boundary_dof(threshold = is_outflow_boundary)
    is_Tin_bdof = space2.is_boundary_dof(threshold = is_inflow_boundary)
    is_Twall_bdof = space2.is_boundary_dof(threshold = is_wall_boundary)
    
    is_T_bdof[is_Tout_bdof] = False 
    xT[is_Tin_bdof] = 525
    xT[is_Twall_bdof] = 323
    

    isBdDof = np.hstack([is_ux_bdof,is_uy_bdof,is_p_bdof,is_T_bdof])
    xx = np.hstack([xux, xuy, xp, xT])
     
    bdIdx = np.zeros(gdof, dtype=np.int_)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    for zz in range(maxit):
        D0 = assemble.matrix([udegree,1], [udegree,0], rhofun(bcs)*us(bcs)[...,0])
        D1 = assemble.matrix([udegree,2], [udegree,0], rhofun(bcs)*us(bcs)[...,1])
        '''
        grad_Ts = Ts.grad_value(bcs) 
        DT0 = assemble.matrix([udegree,0], [tdegree, 0], grad_Ts[...,0]*rhofun(bcs)*Cfun(bcs)) 
        DT1 = assemble.matrix([udegree,0], [tdegree, 0], grad_Ts[...,1]*rhofun(bcs)*Cfun(bcs)) 
        A = bmat([[1/dt*M + S+D0+D1, None, -Sx, None],\
                [None, 1/dt*M + S +D0+D1, -Sy, None],\
                [Sx.T, Sy.T, None, None],\
                [DT0, DT1, None, 1/dt*MT+ST]], format='csr')
        '''
        
        DT0 = assemble.matrix([tdegree,1], [tdegree, 0], us(bcs)[...,0]*rhofun(bcs)*Cfun(bcs)) 
        DT1 = assemble.matrix([tdegree,2], [tdegree, 0], us(bcs)[...,1]*rhofun(bcs)*Cfun(bcs)) 
        A = bmat([[1/dt*M + S+D0+D1, None, -Sx, None],\
                [None, 1/dt*M + S +D0+D1, -Sy, None],\
                [Sx.T, Sy.T, None, None],\
                [None, None, None, 1/dt*MT+ST+DT0+DT1]], format='csr')
        
        A = T@A + Tbd
        b[isBdDof] = xx[isBdDof]
        ctx.set_centralized_sparse(A)
        x = b.copy()
        ctx.set_rhs(x)
        ctx.run(job=6)
        
        u1[:,0] = x[0:gdof2]
        u1[:,1] = x[gdof2:2*gdof2]
        p1[:] = x[2*gdof2:-gdof2]
        T1[:] = x[-gdof2:]
    
        us[:] = u1 
        Ts[:] = T1 
    

    phi0 = level_set(dt, phi0, u1)     
    if i%5 == 0 and i!=0:
        cellscale = np.sqrt(np.min(mesh.entity_measure('cell')))
        phi0 = rein(phi0, cellscale)
    # 网格细化
    for j in range(5): 
        cell2dof2 = space2.cell_to_dof()
        cell2dof1 = space1.cell_to_dof()
        phi0c2f = phi0[cell2dof2]
        u1xc2f = u1[:,0][cell2dof2]
        u1yc2f = u1[:,1][cell2dof2]
        p1c2f = p1[cell2dof1]
        T1c2f = T1[cell2dof1]
        cellmeasure = mesh.entity_measure('cell')
        isMark = np.abs(np.mean(phi0c2f,axis=-1))<epsilon
        isMark = np.logical_and(np.mean(phi0c2f,axis=-1)>-0.01,isMark)
        isMark = np.logical_and(isMark,cellmeasure>4e-5)
        data = {'phi0':phi0c2f,'u1x':u1xc2f,'u1y':u1yc2f,'p1':p1c2f,'T1':T1c2f} 
        option = mesh.bisect_options(data=data,disp=False)
        mesh.bisect(isMark,options=option)

        space2 = LagrangeFiniteElementSpace(mesh, p=2)
        space1 = LagrangeFiniteElementSpace(mesh, p=1)
        cell2dof2 = space2.cell_to_dof()
        cell2dof1 = space1.cell_to_dof()
        phi0 = space2.function()
        u1 = space2.function(dim=2)
        p1 = space1.function()
        T1 = space2.function()
        phi0[cell2dof2.reshape(-1)] = option['data']['phi0'].reshape(-1)
        u1[:,0][cell2dof2.reshape(-1)] = option['data']['u1x'].reshape(-1)
        u1[:,1][cell2dof2.reshape(-1)] = option['data']['u1y'].reshape(-1)
        p1[cell2dof1.reshape(-1)] = option['data']['p1'].reshape(-1)
        T1[cell2dof1.reshape(-1)] = option['data']['T1'].reshape(-1)
    
    
    #网格粗化
    for j in range(5):
        cell2dof2 = space2.cell_to_dof()
        cell2dof1 = space1.cell_to_dof()
        phi0c2f = phi0[cell2dof2]
        u1xc2f = u1[:,0][cell2dof2]
        u1yc2f = u1[:,1][cell2dof2]
        p1c2f = p1[cell2dof1]
        T1c2f = T1[cell2dof1]
        cellmeasure = mesh.entity_measure('cell')
        isMark = np.abs(np.mean(phi0c2f,axis=-1))>epsilon
        isMark = np.logical_and(np.mean(phi0c2f,axis=-1)<0.01,isMark)
        data = {'phi0':phi0c2f,'u1x':u1xc2f,'u1y':u1yc2f,'p1':p1c2f,'T1':T1c2f} 
        option = mesh.bisect_options(data=data,disp=False)
        mesh.coarsen(isMark,options=option)

        space2 = LagrangeFiniteElementSpace(mesh, p=2)
        space1 = LagrangeFiniteElementSpace(mesh, p=1)
        cell2dof2 = space2.cell_to_dof()
        cell2dof1 = space1.cell_to_dof()
        phi0 = space2.function()
        u1 = space2.function(dim=2)
        p1 = space1.function()
        T1 = space2.function()
        phi0[cell2dof2.reshape(-1)] = option['data']['phi0'].reshape(-1)
        u1[:,0][cell2dof2.reshape(-1)] = option['data']['u1x'].reshape(-1)
        u1[:,1][cell2dof2.reshape(-1)] = option['data']['u1y'].reshape(-1)
        p1[cell2dof1.reshape(-1)] = option['data']['p1'].reshape(-1)
        T1[cell2dof1.reshape(-1)] = option['data']['T1'].reshape(-1)

    
    
    u0 = space2.function(dim=2) 
    p0 = space1.function() 
    T0 = space2.function() 
    u0[:] = u1 
    p0[:] = p1
    T0[:] = T1
    us = space2.function(dim=2)
    Ts = space2.function()
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
