import numpy as np
import time
import pandas as pd
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, coo_matrix
import copy 

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.pde.nonlinear_parabolic2d import pLaplaceData as Pde 

NNN = 4
pde = Pde(n=NNN)

N = 300
T = 1
t = np.linspace(0, T, N+1)
t = np.cumsum(t)
t = T*t/t[-1]

mesh = pde.init_mesh(n=3)
space = LagrangeFiniteElementSpace(mesh, p=1)
gdof = space.dof.number_of_global_dofs()

bcs, ws = space.integrator.get_quadrature_points_and_weights()
gphi = space.grad_basis(bcs) #(NQ, NC, ldof, GD)
phi = space.basis(bcs) #(NQ, NC, ldof, GD)
pp = space.mesh.bc_to_point(bcs) #(NQ, NC, GD)
c2d = space.dof.cell_to_dof()
cm = space.mesh.entity_measure('cell')

def B_matrix(space, a_u, uk):
    """!
    @brief 
    """
    a_u_val = a_u(bcs) #(NQ, NC, GD)

    #uh0[c2d]: (NC, ldof)   ghi: (NQ, NC, ldof, GD)
    uk_gval = uk.grad_value(bcs)
    B = np.einsum("qcld, qcd, qcmg, qcg, q, c->cml", gphi, uk_gval, gphi,
            a_u_val, ws, cm)

    I = np.broadcast_to(c2d[:, :, None], shape=B.shape)
    J = np.broadcast_to(c2d[:, None, :], shape=B.shape)
    B = csr_matrix((B.flat, (I.flat, J.flat)), shape=(gdof, gdof))
    return B

#质量矩阵和刚度矩阵
M = space.mass_matrix()

#初始解
uh = [space.function() for i in range(N+1)] 
uh[0] = space.interpolation(pde.init_solution)
for i in range(1, N+1):
    uh[i][:] = uh[i-1][:].copy()
    F0 = M@(uh[i-1][:])

    tau = t[i]-t[i-1]

    #边界条件
    BC = DirichletBC(space, pde.dirichlet)
    while(True):
        S = space.stiff_matrix(c=pde.a(uh[i]))
        B = B_matrix(space, pde.a_u(uh[i]), uh[i]) 
        F1 = M@(uh[i][:])
        F2 = tau*S@(uh[i][:])
        A = M + tau*S + tau*B
        F = F0-F1-F2
        A, F = BC.apply(A, F)

        dU = spsolve(A, F)
        uh[i][:] = uh[i][:] + dU
        print("i = ", i, "err = ", np.max(np.abs(dU)), np.max(abs(uh[i])))
        if(np.max(np.abs(dU))<1e-12):
            break

# vtk
mesh3d = copy.deepcopy(mesh)
node = mesh3d.entity("node")
node = np.c_[node, np.zeros((len(node), 1), dtype=np.float_)]
mesh3d.node = node
node = mesh3d.entity("node")
for i in range(N):
    node[:, -1] = uh[i][:]
    mesh3d.nodedata['val'] = uh[i][:]
    #mesh.to_vtk(fname="../../../study/data/s%i.vtu"%i)
    fname = "../../../study/data/mesh3d"+('%i'%i).zfill(4)+'.vtu'
    mesh3d.to_vtk(fname=fname)

    mesh.nodedata['val'] = uh[i][:]
    fname = "../../../study/data/mesh"+('%i'%i).zfill(4)+'.vtu'
    mesh.to_vtk(fname=fname)


