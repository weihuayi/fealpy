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
        ScalarMassIntegrator, PressWorkIntegrator, 
        PressWorkIntegrator1, ScalarConvectionIntegrator)
from fealpy.experimental.fem import LinearForm, ScalarSourceIntegrator
from fealpy.experimental.fem import DirichletBC
from fealpy.experimental.sparse.linalg import sparse_cg
from fealpy.experimental.sparse import COOTensor

from fealpy.pde.navier_stokes_equation_2d import FlowPastCylinder
from fealpy.decorator import barycentric, cartesian
from fealpy.timeintegratoralg import UniformTimeLine

backend = 'numpy'
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

omesh = pde.mesh1(0.01)
node = bm.from_numpy(omesh.entity('node'))
cell = bm.from_numpy(omesh.entity('cell'))
mesh = TriangleMesh(node, cell)


timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

pspace = LagrangeFESpace(mesh, p=pdegree)
uspace = LagrangeFESpace(mesh, p=udegree)


u0x = uspace.function()
u0y = uspace.function()
u0 = bm.stack((u0x, u0y), axis=1)
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

bform = BilinearForm(uspace)
bform.add_integrator(ScalarMassIntegrator(rho/dt, q=q))
M = bform.assembly()

bform = BilinearForm(uspace)
bform.add_integrator(ScalarDiffusionIntegrator(mu, q=q))
S = bform.assembly()

bform = BilinearForm((pspace, uspace))
bform.add_integrator(PressWorkIntegrator(q=q)) 
APX = bform.assembly()

bform = BilinearForm((pspace, uspace))
bform.add_integrator(PressWorkIntegrator1(q=q)) 
APY = bform.assembly()

#边界处理
xx = bm.zeros(gdof)
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
ipoint = uspace.interpolation_points()[u_isbddof_in]
uinfow = pde.u_inflow_dirichlet(ipoint)
xx[0:ugdof][u_isbddof_in] = uinfow[:,0]
xx[ugdof:2*ugdof][u_isbddof_in] = uinfow[:,1]

p_isBdDof_p0 = pspace.is_boundary_dof(threshold = pde.is_outflow_boundary) 
xx[2*ugdof:][p_isBdDof_p0] = 0 
isBdDof = bm.concatenate([u_isbddof, u_isbddof, p_isBdDof_p0], axis=0)

for i in range(1):
    t1 = timeline.next_time_level()
    print("time=", t1)
    
    @barycentric
    def concoef(bcs, index):
        a1 = u0x(bcs, index)
        a2 = u0y(bcs, index)
        result = bm.concatenate((a1[...,bm.newaxis],a2[..., bm.newaxis]), axis=2)
        return result
    
    bform = BilinearForm(uspace)
    bform.add_integrator(ScalarConvectionIntegrator(concoef, q=4))
    C = bform.assembly() 
    
    indices = bm.tensor([[],[]])
    data = bm.tensor([])
    zeros_0 = COOTensor(indices, data, (ugdof,ugdof))
    zeros_1 = COOTensor(indices, data, (pgdof,pgdof))
    
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
        A0 = bm.concatenate([M+S+C, zeros_0, -APX], axis=1)
        A1 = bm.concatenate([zeros_0, M+S+C, -APY], axis=1)
        A2 = bm.concatenate([-APX.T, -APY.T ,zeros_1], axis=1)
        A = bm.concatenate((A0,A1,A2),axis=0)

    b0 = M@u0[:,0] 
    b1 = M@u0[:,1]
    b2 = bm.zeros(pgdof) 
    b = bm.concatenate([b0,b1,b2])
    
    b -= A@xx
    b[isBdDof] = xx[isBdDof]
    
    kwargs = A.values_context()
    indices = A.indices()
    new_values = bm.copy(A.values())
    IDX = isBdDof[indices[0, :]] | isBdDof[indices[1, :]]
    new_values[IDX] = 0
    A = COOTensor(indices, new_values, A.sparse_shape)
 
    index, = bm.nonzero(isBdDof, as_tuple=True)
    shape = new_values.shape[:-1] + (len(index), )
    one_values = bm.ones(shape, **kwargs)
    one_indices = bm.stack([index, index], axis=0)
    A1 = COOTensor(one_indices, one_values, A.sparse_shape)
    A = A.add(A1).coalesce()
    
    ''' 
    import scipy.sparse as sp
    values = A.values()
    indices = A.indices()
    A = sp.coo_matrix((values, (indices[0], indices[1])), shape=A.shape) 
    A = A.tocsr()
    x = sp.linalg.spsolve(A,b)
    '''
    x = sparse_cg(A, b, maxiter=10000)

    u1x[:] = x[:ugdof]
    u1y[:] = x[ugdof:2*ugdof]
    p1[:] = x[2*ugdof:]
     
    u0x[:] = u1x
    u0y[:] = u1y
    u0 = bm.stack((u0x, u0y), axis=1)
    
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['velocity'] = u0
    mesh.nodedata['pressure'] = p1
    mesh.to_vtk(fname=fname)
    
    timeline.advance() 
