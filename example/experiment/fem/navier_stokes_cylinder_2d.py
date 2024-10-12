#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: navier_stokes_cylinder_2d.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Mon 12 Aug 2024 04:52:25 PM CST
	@bref 
	@ref 
'''  
from fealpy.experimental import logger
logger.setLevel('ERROR')

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.functionspace import TensorFunctionSpace
from fealpy.experimental.fem import (
        BilinearForm, ScalarDiffusionIntegrator, 
        ScalarMassIntegrator, PressWorkIntegrator0, PressWorkIntegrator ,
        PressWorkIntegrator1, ScalarConvectionIntegrator)


from fealpy.experimental.fem import LinearForm, ScalarSourceIntegrator
from fealpy.experimental.fem import DirichletBC
from fealpy.experimental.sparse import COOTensor
from fealpy.experimental.fem import VectorSourceIntegrator
from fealpy.experimental.fem import BlockForm
from fealpy.experimental.solver import cg 

from fealpy.pde.navier_stokes_equation_2d import FlowPastCylinder
from fealpy.decorator import barycentric, cartesian
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.experimental.fem import DirichletBC

backend = 'pytorch'
#backend = 'numpy'
bm.set_backend(backend)

output = './'
udegree = 2
pdegree = 1
q = 4
T = 5
nt = 5000
pde = FlowPastCylinder()
rho = pde.rho
mu = pde.mu

omesh = pde.mesh1(0.05)
node = bm.from_numpy(omesh.entity('node'))
cell = bm.from_numpy(omesh.entity('cell'))
mesh = TriangleMesh(node, cell)


timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

pspace = LagrangeFESpace(mesh, p=pdegree)
uspace = LagrangeFESpace(mesh, p=udegree)
tensor_uspace = TensorFunctionSpace(uspace, (-1, 2))

u0x = uspace.function()
u0y = uspace.function()
u0 = bm.stack((u0x[:], u0y[:]), axis=1)
u1x = uspace.function()
u1y = uspace.function()
p1 = pspace.function()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
gdof = pgdof+2*ugdof

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['velocity'] = u0 
mesh.nodedata['pressure'] = p1
mesh.to_vtk(fname=fname)
'''
M_bform = BilinearForm(uspace)
M_bform.add_integrator(ScalarMassIntegrator(rho/dt, q=q))
M = M_bform.assembly()
'''

'''
from scipy.sparse import coo_array, bmat
def coo(A):
    data = A._values
    indices = A._indices
    return coo_array((data, indices))

A = bmat([[coo(M), None],[None, coo(M)]],  format='coo')

print(bm.sum(bm.abs(A.toarray()-coo(MM).toarray())))
exit()

S_bform = BilinearForm(uspace)
S_bform.add_integrator(ScalarDiffusionIntegrator(mu, q=q))
S = S_bform.assembly()

P_bform = BilinearForm((pspace, tensor_uspace))
P_bform.add_integrator(PressWorkIntegrator(mu, q=q))
P = P_bform.assembly()
'''

A_bform = BilinearForm(uspace)
A_bform.add_integrator(ScalarMassIntegrator(rho/dt, q=q))
A_bform.add_integrator(ScalarDiffusionIntegrator(mu, q=q)) 
ConvectionIntegrator = ScalarConvectionIntegrator(q=q)
A_bform.add_integrator(ConvectionIntegrator)

APX_bform = BilinearForm((pspace, uspace))
APX_bform.add_integrator(PressWorkIntegrator0(coef=-1, q=q)) 
APX = APX_bform.assembly()

APY_bform = BilinearForm((pspace, uspace))
APY_bform.add_integrator(PressWorkIntegrator1(coef=-1, q=q)) 
APY = APY_bform.assembly()

#边界处理
xx = bm.zeros(gdof, dtype=mesh.ftype)
u_isbddof_u0 = uspace.is_boundary_dof()
u_isbddof_in = uspace.is_boundary_dof(threshold = pde.is_inflow_boundary)
u_isbddof_out = uspace.is_boundary_dof(threshold = pde.is_outflow_boundary)
u_isbddof_circle = uspace.is_boundary_dof(threshold = pde.is_circle_boundary)

u_isbddof_u0[u_isbddof_in] = False 
u_isbddof_u0[u_isbddof_out] = False 
xx[0:ugdof][u_isbddof_u0] = 0
xx[ugdof:2*ugdof][u_isbddof_u0] = 0

u_isbddof = u_isbddof_u0
u_isbddof[u_isbddof_in] = True
#ipoint = uspace.interpolation_points()[u_isbddof_in]
ipoint = uspace.interpolation_points()
uinflow = pde.u_inflow_dirichlet(ipoint)

#xx[0:ugdof][u_isbddof_in] = uinflow[:,0]
#xx[ugdof:2*ugdof][u_isbddof_in] = uinflow[:,1]
#print(u_bd_in.shape)
#print(xx.shape)
#xx[u_bd_in] = uinflow.reshape(-1)

p_isBdDof_p0 = pspace.is_boundary_dof(threshold = pde.is_outflow_boundary) 
bd = bm.concatenate((u_isbddof_in, u_isbddof_in, p_isBdDof_p0))
value_bd = bm.concatenate((uinflow[:,0],uinflow[:,1], bm.zeros(pgdof)))
xx[bd] = value_bd[bd] 


isBdDof = bm.concatenate([u_isbddof, u_isbddof, p_isBdDof_p0], axis=0)

for i in range(10):
    t1 = timeline.next_time_level()
    print("time=", t1)
    
    @barycentric
    def concoef(bcs, index):
        a1 = u0x(bcs, index)
        a2 = u0y(bcs, index)
        result = bm.concatenate((a1[...,bm.newaxis],a2[..., bm.newaxis]), axis=2)
        return result
    

    ConvectionIntegrator.coef = concoef
    ConvectionIntegrator.clear()

    A = BlockForm([[A_bform, None, APX_bform],
                  [None, A_bform, APY_bform],
                   [APX_bform.T, APY_bform.T, None]])
    A = A.assembly()
    '''
    if backend == 'numpy':
        from scipy.sparse import coo_array, bmat
        def coo(A):
            data = A._values
            indices = A._indices
            return coo_array((data, indices))

        A = bmat([[coo(M+S+C), None,coo(-APX)],
                [None, coo(M+S+C), coo(-APY)],
                [coo(-APX).T, coo(-APY).T, None]], format='coo')
        A = COOTensor(bm.stack([A.row,A.col],axis=0), A.data, spshape=A.shape)
        A = A.coalesce()
    
    if backend == 'pytorch':
        indices = bm.tensor([[],[]])
        data = bm.tensor([])
        zeros_0 = COOTensor(indices, data, (ugdof,ugdof))
        zeros_1 = COOTensor(indices, data, (pgdof,pgdof))
        A0 = bm.concatenate([M+S+C, zeros_0, -APX], axis=1)
        A1 = bm.concatenate([zeros_0, M+S+C, -APY], axis=1)
        A2 = bm.concatenate([-APX.T, -APY.T ,zeros_1], axis=1)
        A = bm.concatenate((A0,A1,A2),axis=0)
    '''
    lform = LinearForm(uspace)
    lform.add_integrator(ScalarSourceIntegrator(u0x))
    b0 = lform.assembly()
    lform = LinearForm(uspace)
    lform.add_integrator(ScalarSourceIntegrator(u0y))
    b1 = lform.assembly()
    b2 = bm.zeros(pgdof) 
    b = bm.concatenate([b0,b1,b2])
    
    print(xx.dtype)
    print(A.values().dtype)
    b -= A@xx
    b[isBdDof] = xx[isBdDof]
    
    A = DirichletBC(uspace, xx, isDDof=isBdDof).apply_matrix(A, check=False)
    #A,b = DirichletBC(uspace, xx, isDDof=isBdDof).apply(A, b, check=None)

    import scipy.sparse as sp
    values = A.values()
    indices = A.indices()
    A = sp.coo_matrix((values, (indices[0], indices[1])), shape=A.shape) 
    A = A.tocsr()
    x = sp.linalg.spsolve(A,b)
    
    x = bm.array(x)
    ''' 
    x = cg(A, b, maxiter=10000)
    '''
    
    u1x[:] = x[:ugdof]
    u1y[:] = x[ugdof:2*ugdof]
    p1[:] = x[2*ugdof:]
     
    u0x[:] = u1x[:]
    u0y[:] = u1y[:]
    u0 = bm.stack((u0x[:], u0y[:]), axis=1)
    
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['velocity'] = u0
    mesh.nodedata['pressure'] = p1
    mesh.to_vtk(fname=fname)
    
    timeline.advance() 
