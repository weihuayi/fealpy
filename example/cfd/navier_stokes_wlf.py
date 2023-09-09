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


def rein(phi0, cellscale, alpha=None, dt=0.0001, eps=1e-4, nt=4):
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
        print(error) 
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
nt = 1500
ns = 10
h = 1
domain = [0,10*h,0,h]
mesh = TriangleMesh.from_box(box=domain, nx=10*ns, ny=ns)
dx = 1/ns
tmesh = UniformTimeLine(0,T,nt)
dt = tmesh.dt
eps = 1e-12

wrho = 1020
grho = 1.02
wC = 1700
gC = 0.588*wC
wlambda = 0.173
glambda = 0.139*wlambda



fname = './' + 'test_'+ '.vtu'

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
    val =  (x-h) - 2*h*y*(h-y)/h**2
    #val =  x
    return val
           
@cartesian
def u_inflow_dirichlet(p):
    x = p[...,0]
    y = p[...,1]
    value = np.zeros(p.shape)
    value[...,0] = 5
    value[...,1] = 0
    return value


@cartesian
def usolution(p):
    x = p[...,0]
    y = p[...,1]
    u = np.zeros(p.shape)
    u[...,0] = 4*y*(1-y)
    return u
'''
如何把梯度写成函数
def crosswlf(T,P,u):
    result = T.space.function()
    eta0 = 1.9e11*np.exp(-27.396*(T-417.15)/(51.6+(T-417.15)))
    deformnation = u.grad_value(bcs)
    deformnation = 1/2*(deformnation + deformnation.transpose(0,2,1,3))
    print(deformnation.shape)
    gamma = np.sqrt(2*np.einsum('ijkl,ijkl->ij',deformnation,deformnation))
    result[:] = eta0/(1+eta0)
    return  result
'''

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
    isMark = np.abs(np.mean(phi0c2f,axis=-1))< 0.05
    data = {'phi0':phi0c2f} 
    option = mesh.bisect_options(data=data, disp=False)
    mesh.bisect(isMark,options=option)

    space2 = LagrangeFESpace(mesh,p=2, doforder='sdofs')
    space1 = LagrangeFESpace(mesh,p=1)
    cell2dof = space2.cell_to_dof()
    phi0 = space2.function()
    phi0[cell2dof.reshape(-1)] = option['data']['phi0'].reshape(-1)


## 初始网格
#u0 = space2.function(dim=2)
u0 = space2.interpolate(usolution, dim=2)
us = space2.function(dim=2)
u1 = space2.function(dim=2)
p0 = space1.function()
ps = space1.function()
p1 = space1.function()
T0 = space1.function()
Ts = space1.function()
T1 = space1.function()


rhofun =  heaviside(phi0, dx, wrho, grho)
Cfun =  heaviside(phi0, dx, wrho, grho)
etafun =  heaviside(phi0, dx, wrho, grho)
lambdafun =  heaviside(phi0, dx, wrho, grho)


mesh.nodedata['velocity'] = u1
mesh.nodedata['pressure'] = p1
mesh.nodedata['tempture'] = T1
mesh.nodedata['rho'] = rhofun
mesh.nodedata['surface'] = phi0

mesh.to_vtk(fname=fname)



for i in range(0, 1):
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)
    
    us[:] = u0
    Ts[:] = T0

    for i in range(1):
        # 动量方程矩阵
        BF0 = BilinearForm(space2)        
        @barycentric 
        def dt_rho(bcs, index):
            return 1/dt*rhofun(bcs, index)
        BF0.add_domain_integrator(ScalarMassIntegrator(dt_rho))
        
        @barycentric 
        def rho_u(bcs,index):
            result = np.einsum('ik,ijk->ijk',rhofun(bcs,index),us(bcs,index))
            return result
        BF0.add_domain_integrator(ScalarConvectionIntegrator(rho_u))
        BF0.add_domain_integrator(ScalarDiffusionIntegrator(etafun)) 
        AU = BF0.assembly() 
        '''
        SS = nuspace.stiff_matrix(etafun)
        MM = nuspace.mass_matrix(dt_rho)
        CC = nuspace.convection_matrix(rho_u)
        AA = SS+MM+CC
        '''
        # 连续方程矩阵 
        BF1 = MixedBilinearForm((space1,), 2*(space2,)) 
        BF1.add_domain_integrator(PressWorkIntegrator()) 
        AP = BF1.assembly()
        
        BF2 = BilinearForm(space1)
        BF2.add_domain_integrator(ScalarDiffusionIntegrator(etafun))
        ASP = BF2.assembly() 
        
        
        # 能量方程矩阵
        BF3 = BilinearForm(space1)
        @barycentric 
        def dt_rho_C(bcs,index):
            return 1/dt*rhofun(bcs,index)*Cfun(bcs,index)
        BF3.add_domain_integrator(ScalarMassIntegrator(dt_rho_C))
        BF3.add_domain_integrator(ScalarDiffusionIntegrator(lambdafun)) 
        
        @barycentric 
        def rho_C_u(bcs,index):
            result = rhofun(bcs,index)*Cfun(bcs,index)
            result = np.einsum('ik,ijk->ijk',result,us(bcs,index))
            return result
        BF3.add_domain_integrator(ScalarConvectionIntegrator(c=rho_C_u))
        AT = BF3.assembly() 
        
        A = bmat([[AU, None],[None, AU]], format='csr') 
        A = bmat([[A,  -AP, None],\
                [AP.T, 1e-8*ASP, None],\
                [None,None,AT]], format='csr')
        
        # 右端向量
        LFU = LinearForm(2*(space2,))
        LFU.add_domain_integrator(VectorSourceIntegrator(u0))
        bu = LFU.assembly()
        bp = np.zeros(space1.number_of_global_dofs())
        
        @barycentric
        def bT_source(bcs, index=np.s_[:]):
            val0 = 1/dt * T0(bcs, index) 
            gradu = u0.grad_value(bcs, index)
            D = gradu + gradu.transpose(0,2,1,3)
            etaD = np.einsum('il,ijkl->ijkl',etafun(bcs, index), D)
            etaD[:,0,0,:] +=p0(bcs, index)
            etaD[:,1,1,:] +=p0(bcs, index)
            val1 = np.einsum('ijkc,ijkc->ic',etaD, gradu)
            return val1

        
        LFT = LinearForm(space1)
        LFT.add_domain_integrator(ScalarSourceIntegrator(bT_source))
        bT = LFT.assembly() 

        b = np.hstack([bu,bp,bT])
        # 边界处理
        is_u_bdof = space2.is_boundary_dof()
        is_uwall_bdof = space2.is_boundary_dof(threshold=is_wall_boundary)
        #is_ustick_bdof = space2.is_boundary_dof(threshold = is_stick_boundary)
        is_uin_bdof = space2.is_boundary_dof(threshold = is_inflow_boundary)
        #is_uout_bdof = space2.is_boundary_dof(threshold = is_outflow_boundary)
        #is_p_bdof = space1.is_boundary_dof(threshold = is_outflow_boundary)


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
