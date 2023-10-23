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


T = 5
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

wrho = 1020
grho = wrho*0.001
wC = 1700
gC = 0.588*wC
wlambda = 0.173
glambda = 0.139*wlambda
geta = 1.792e-5
weta = geta/0.0001
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

space2 = LagrangeFESpace(mesh, p=2, doforder='sdofs')
space1 = LagrangeFESpace(mesh, p=1)

nuspace = LagrangeFiniteElementSpace(mesh, p=2)
npspace = LagrangeFiniteElementSpace(mesh, p=1)
ntspace = LagrangeFiniteElementSpace(mesh, p=1)

## 加密
phi0 = space2.interpolate(dist)
for i in range(5):
    cell2dof = mesh.cell_to_ipoint(2)
    phi0c2f = phi0[cell2dof]
    isMark = np.abs(np.mean(phi0c2f,axis=-1))< epsilon
    data = {'phi0':phi0c2f} 
    option = mesh.bisect_options(data=data, disp=False)
    mesh.bisect(isMark,options=option)

    space2 = LagrangeFESpace(mesh, p=2, doforder='sdofs')
    space1 = LagrangeFESpace(mesh,p=1)
    cell2dof = space2.cell_to_dof()
    phi0 = space2.function()
    phi0[cell2dof.reshape(-1)] = option['data']['phi0'].reshape(-1)


## 初始网格
u0 = space2.function(dim=2)
#u0 = space2.interpolate(usolution, dim=2)
us = space2.function(dim=2)
u1 = space2.function(dim=2)
p0 = space1.function()
p1 = space1.function()
T0 = space1.function()
T1 = space1.function()
phi0 = space2.interpolate(dist)

rhofun =  heaviside(phi0, epsilon, wrho, grho)
Cfun =  heaviside(phi0, epsilon, wC, gC)
lambdafun =  heaviside(phi0, epsilon, wlambda, glambda)
etafun2 =  heaviside(phi0, epsilon, weta, geta)
mesh.nodedata['velocity'] = u0.T
mesh.nodedata['pressure'] = p0
mesh.nodedata['tempture'] = T0
mesh.nodedata['rho'] = rhofun
mesh.nodedata['surface'] = phi0
mesh.nodedata['比热容'] = Cfun
mesh.nodedata['热扩散系数'] = lambdafun
mesh.nodedata['粘度系数'] = etafun2

fname = output + 'test_0000000000.vtu'
#fname = output + 'test_.vtu'
mesh.to_vtk(fname=fname)
ctx = DMumpsContext()
ctx.set_silent()

for i in range(3):
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)
     
    gdof2 = space2.number_of_global_dofs()
    gdof1 = space1.number_of_global_dofs()
    gdof = 2*gdof2 + 2*gdof1
    us[:] = u0
    
    # 动量方程矩阵
    BF0 = BilinearForm(space2)        
    
    @barycentric 
    def dt_rho(bcs, index):
        return 1/dt*rhofun(bcs, index)
    BF0.add_domain_integrator(ScalarMassIntegrator(dt_rho)) 
    
    @barycentric
    def etafun(bcs, index):
        # crosswlf
        eta0 = 1.9e11*np.exp(-27.396*(T0(bcs, index)-417.15)/(51.6+(T0(bcs, index)-417.15)))
        deformnation = u0.grad_value(bcs, index)
        deformnation = 1/2*(deformnation + deformnation.transpose(0,2,1,3))
        gamma = np.sqrt(2*np.einsum('ijkl,ikjl->il',deformnation,deformnation))
        result = eta0/(1+(eta0*gamma/182680)**(1-0.574))
        #heaviside
        pbcs = phi0(bcs,index) 
        tag = (-epsilon<= pbcs)  & (pbcs <= epsilon)
        tag1 = pbcs > epsilon
        result[tag1] = geta
        result[tag] = 0.5*(1+pbcs[tag]/epsilon) 
        result[tag] += 0.5*np.sin(np.pi*pbcs[tag]/epsilon)/np.pi
        return  result
        
    
    @barycentric
    def Reetafun(bcs, index):
        return etafun2(bcs, index)/Re
    BF0.add_domain_integrator(ScalarDiffusionIntegrator(Reetafun)) 
    AU = BF0.assembly() 
    
    # 连续方程矩阵 
    BF1 = MixedBilinearForm((space1,), 2*(space2,)) 
    BF1.add_domain_integrator(PressWorkIntegrator()) 
    AP = BF1.assembly()
    
    BF2 = BilinearForm(space1)
    BF2.add_domain_integrator(ScalarDiffusionIntegrator())
    ASP = BF2.assembly()  
    
    # 能量方程矩阵
    BF3 = BilinearForm(space1)
    
    @barycentric 
    def dt_rho_C(bcs,index):
        return Pe*rhofun(bcs,index)*Cfun(bcs,index) /dt
    BF3.add_domain_integrator(ScalarMassIntegrator(dt_rho_C))
    BF3.add_domain_integrator(ScalarDiffusionIntegrator(lambdafun)) 
    
    AT = BF3.assembly() 
    
    # 右端向量
    LFU = LinearForm(2*(space2,))
    
    @barycentric 
    def dt_rho_u0(bcs, index):
        result = np.einsum('ik, ijk->ijk', rhofun(bcs, index), u0(bcs, index))
        return result/dt
    
    LFU.add_domain_integrator(VectorSourceIntegrator(dt_rho_u0))
    bu = LFU.assembly()
    bp = np.zeros(space1.number_of_global_dofs())
    ## 能量方程右端向量 
    LFT = LinearForm(space1) 
    @barycentric
    def bT_source(bcs, index=np.s_[:]):
        val0 = 1/dt * T0(bcs, index) 
        gradu = u0.grad_value(bcs, index)
        D = gradu + gradu.transpose(0,2,1,3)
        etaD = np.einsum('il,ijkl->ijkl',etafun2(bcs, index), D)
        etaD[:,0,0,:] +=p0(bcs, index)
        etaD[:,1,1,:] +=p0(bcs, index)
        val1 = np.einsum('ijkc,ijkc->ic',etaD, gradu)
        return Br*val1
    LFT.add_domain_integrator(ScalarSourceIntegrator(bT_source)) 
    
    @barycentric 
    def dt_rho_C_T0(bcs,index):
        return Pe*rhofun(bcs,index)*Cfun(bcs,index)*T0(bcs,index)/dt 
    LFT.add_domain_integrator(ScalarSourceIntegrator(dt_rho_C_T0))
    bT = LFT.assembly() 
    b = np.hstack([bu,bp,bT])
    
    # 边界处理
    is_ux_bdof = space2.is_boundary_dof()
    is_uy_bdof = space2.is_boundary_dof()
    is_p_bdof = space1.is_boundary_dof()
    is_T_bdof = space1.is_boundary_dof()
    xux = np.zeros_like(is_ux_bdof, np.float64)
    xuy = np.zeros_like(is_uy_bdof, np.float64)
    xp = np.zeros_like(is_p_bdof, np.float64)
    xT = np.zeros_like(is_T_bdof, np.float64)

    ## 速度边界处理  
    is_uout_bdof = space2.is_boundary_dof(threshold = is_outflow_boundary)
    is_uin_bdof = space2.is_boundary_dof(threshold = is_inflow_boundary)
    
    is_ux_bdof[is_uout_bdof] = False 
    is_uy_bdof[is_uout_bdof] = False 
    
    xux[is_uin_bdof] = 5
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
    
    is_p_bdof[is_pmelt_bdof] = False 
    #xp[is_pin_bdof] = 8
    #xp[is_pout_bdof] = 0

    ## 温度边界条件处理   
    is_Tout_bdof = space1.is_boundary_dof(threshold = is_outflow_boundary)
    is_Tin_bdof = space1.is_boundary_dof(threshold = is_inflow_boundary)
    is_Twall_bdof = space1.is_boundary_dof(threshold = is_wall_boundary)
    
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
        BF0C = BilinearForm(space2)

        @barycentric 
        def rho_u(bcs,index):
            result = np.einsum('ik,ijk->ijk',rhofun(bcs,index),us(bcs,index))
            return result
        BF0C.add_domain_integrator(ScalarConvectionIntegrator(rho_u))
        AUC = BF0C.assembly()

        '''
        SS = nuspace.stiff_matrix(etafun)
        MM = nuspace.mass_matrix(dt_rho)
        CC = nuspace.convection_matrix(rho_u)
        AA = SS+MM+CC
        '''
        
        BF3C = BilinearForm(space1)
        @barycentric 
        def rho_C_u(bcs,index):
            result = rhofun(bcs,index)*Cfun(bcs,index)
            result = np.einsum('ik,ijk->ijk',result,us(bcs,index))
            return Pe*result
        BF3C.add_domain_integrator(ScalarConvectionIntegrator(c=rho_C_u))
        ATU = BF3C.assembly() 
        
        
        A0 = bmat([[AU+AUC, None],[None, AU+AUC]], format='csr')  
        A = bmat([[A0,  -AP, None],\
                [AP.T, 1e-8*ASP, None],\
                [None,None,AT+ATU]], format='csr')
        
        
        A = T@A + Tbd
        b[isBdDof] = xx[isBdDof]
        ctx.set_centralized_sparse(A)
        x = b.copy()
        ctx.set_rhs(x)
        ctx.run(job=6)
         
        u1[0,:] = x[0:gdof2]
        u1[1,:] = x[gdof2:2*gdof2]
        p1[:] = x[2*gdof2:-gdof1]
        T1[:] = x[-gdof1:]
    
        us[:] = u1 
    
    phi0 = level_set(dt, phi0, u1)     
    # 网格细化
    for j in range(5): 
        cell2dof2 = space2.cell_to_dof()
        cell2dof1 = space1.cell_to_dof()
        phi0c2f = phi0[cell2dof2]
        u1xc2f = u1[0,:][cell2dof2]
        u1yc2f = u1[1,:][cell2dof2]
        p1c2f = p1[cell2dof1]
        T1c2f = T1[cell2dof1]
        cellmeasure = mesh.entity_measure('cell')
        isMark = np.abs(np.mean(phi0c2f,axis=-1))<epsilon
        isMark = np.logical_and(np.mean(phi0c2f,axis=-1)>-0.01,isMark)
        isMark = np.logical_and(isMark,cellmeasure>4e-5)
        data = {'phi0':phi0c2f,'u1x':u1xc2f,'u1y':u1yc2f,'p1':p1c2f,'T1':T1c2f} 
        option = mesh.bisect_options(data=data,disp=False)
        mesh.bisect(isMark,options=option)

        space2 = LagrangeFESpace(mesh, p=2, doforder='sdofs')
        space1 = LagrangeFESpace(mesh,p=1)
        cell2dof2 = space2.cell_to_dof()
        cell2dof1 = space1.cell_to_dof()
        phi0 = space2.function()
        u1 = space2.function(dim=2)
        p1 = space1.function()
        T1 = space1.function()
        phi0[cell2dof2.reshape(-1)] = option['data']['phi0'].reshape(-1)
        u1[0,:][cell2dof2.reshape(-1)] = option['data']['u1x'].reshape(-1)
        u1[1,:][cell2dof2.reshape(-1)] = option['data']['u1y'].reshape(-1)
        p1[cell2dof1.reshape(-1)] = option['data']['p1'].reshape(-1)
        T1[cell2dof1.reshape(-1)] = option['data']['T1'].reshape(-1)
    
    
    #网格粗化
    for j in range(5):
        cell2dof2 = space2.cell_to_dof()
        cell2dof1 = space1.cell_to_dof()
        phi0c2f = phi0[cell2dof2]
        u1xc2f = u1[0,:][cell2dof2]
        u1yc2f = u1[1,:][cell2dof2]
        p1c2f = p1[cell2dof1]
        T1c2f = T1[cell2dof1]
        cellmeasure = mesh.entity_measure('cell')
        isMark = np.abs(np.mean(phi0c2f,axis=-1))>epsilon
        isMark = np.logical_and(np.mean(phi0c2f,axis=-1)<0.01,isMark)
        data = {'phi0':phi0c2f,'u1x':u1xc2f,'u1y':u1yc2f,'p1':p1c2f,'T1':T1c2f} 
        option = mesh.bisect_options(data=data,disp=False)
        mesh.coarsen(isMark,options=option)

        space2 = LagrangeFESpace(mesh, p=2, doforder='sdofs')
        space1 = LagrangeFESpace(mesh,p=1)
        cell2dof2 = space2.cell_to_dof()
        cell2dof1 = space1.cell_to_dof()
        phi0 = space2.function()
        u1 = space2.function(dim=2)
        p1 = space1.function()
        T1 = space1.function()
        phi0[cell2dof2.reshape(-1)] = option['data']['phi0'].reshape(-1)
        u1[0,:][cell2dof2.reshape(-1)] = option['data']['u1x'].reshape(-1)
        u1[1,:][cell2dof2.reshape(-1)] = option['data']['u1y'].reshape(-1)
        p1[cell2dof1.reshape(-1)] = option['data']['p1'].reshape(-1)
        T1[cell2dof1.reshape(-1)] = option['data']['T1'].reshape(-1)

    rhofun =  heaviside(phi0, epsilon, wrho, grho)
    Cfun =  heaviside(phi0, epsilon, wC, gC)
    lambdafun =  heaviside(phi0, epsilon, wlambda, glambda)
    etafun2 =  heaviside(phi0, epsilon, weta, geta)
    #if i%step == 0 and i!=0:
    if i%1 == 0:
        cellscale = np.sqrt(np.min(mesh.entity_measure('cell')))
        #phi0 = rein(phi0, cellscale)
        fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
        mesh.nodedata['velocity'] = u1.T
        mesh.nodedata['pressure'] = p1
        mesh.nodedata['tempture'] = T1
        mesh.nodedata['rho'] = rhofun
        mesh.nodedata['surface'] = phi0
        mesh.nodedata['比热容'] = Cfun
        mesh.nodedata['热扩散系数'] = lambdafun
        mesh.nodedata['粘度系数'] = etafun2
        mesh.to_vtk(fname=fname)
    
    
    u0 = space2.function(dim=2) 
    p0 = space1.function() 
    T0 = space1.function() 
    u0[:] = u1 
    p0[:] = p1
    T0[:] = T1
    us = space2.function(dim=2)
    # 时间步进一层 
    tmesh.advance()
    '''
    mass = nuspace.mass_matrix()
    mass2 = bmat([[mass, None], \
            [None, mass]], format='csr')
    BF12 = BilinearForm(2*(space2,))
    BF12.add_domain_integrator(VectorMassIntegrator())
    
    A1SP = BF12.assembly()
    b00 = A1SP@u0.flatten()
    print(np.sum(np.abs(b00-b0)))
    '''
    '''
    验证ASP
    qf = mesh.integrator(4)
    bcs,ws = qf.get_quadrature_points_and_weights()
    cellmeasure = mesh.entity_measure('cell')
    ucell2dof = nuspace.cell_to_dof()
    pcell2dof = npspace.cell_to_dof()
    gphi = nuspace.grad_basis(bcs)
    phi = npspace.basis(bcs)
    ugdof = nuspace.number_of_global_dofs() 
    pgdof = npspace.number_of_global_dofs() 
    
    ggphi = np.concatenate((gphi[:,:,:,0],gphi[:,:,:,1]),2)

    ugdof = nuspace.number_of_global_dofs()
    S = np.einsum('i,ijk,ijm,j->jkm',ws, gphi[:,:,:,0], phi,cellmeasure)      
    S2 = np.einsum('i,ijk,ijm,j->jkm',ws, gphi[:,:,:,1], phi,cellmeasure)      
    I = np.broadcast_to(ucell2dof[:,:,None],shape = S.shape)
    J = np.broadcast_to(pcell2dof[:,None,:],shape = S.shape)
    S = csr_matrix((S.flat,(I.flat,J.flat)),shape=(ugdof,pgdof))
    S2 = csr_matrix((S2.flat,(I.flat,J.flat)),shape=(ugdof,pgdof))
    SS = np.vstack((S.toarray(),S2.toarray())) 
    print(np.sum(SS-AP.toarray()))
    '''
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
