import numpy as np
from scipy.sparse import csr_matrix
from opt_einsum import contract


from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import timer


def f(p):
    pi = np.pi
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]
    return np.sin(2*pi*x)*np.sin(2*pi*y)*np.sin(2*pi*z)


@timer
def mass_matrix(space):
    qf = space.integrator
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = space.cellmeasure

    phi = space.basis(bcs) # (NQ, 1, ldof)
    M = np.einsum('q, qci, qcj, c->cij', ws, phi, phi, cm, optimize=True) # (NC, ldof, ldof)
    #M = contract('q, qci, qcj, c->cij', ws, phi, phi, cm) # (NC, ldof, ldof)

    #path_info = np.einsum_path('q, qci, qcj, c->cij', ws, phi, phi, cm) # (NC, ldof, ldof)
    #print(path_info[0])
    #print(path_info[1])

    return M


@timer
def stiff_matrix(space):
    qf = space.integrator
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = space.cellmeasure

    gphi = space.grad_basis(bcs) #(NQ, NC, ldof, GD)
    A = np.einsum('q, qcim, qcjm, c->cij', ws, gphi, gphi, cm, optimize=True) # (NC, ldof, ldof)
    #A = contract('q, qcim, qcjm, c->cij', ws, gphi, gphi, cm) # (NC, ldof, ldof)

    #gphi = space.grad_basis(bcs) # (ldof, NQ)
    #A = np.einsum('q, ciqm, cjqm, c->cij', ws, gphi, gphi, cm, optimize=True) # (NC, ldof, ldof)

    #path_info = np.einsum_path('q, qcim, qcjm, c->cij', ws, gphi, gphi, cm) # (NC, ldof, ldof)
    #print(path_info[0])
    #print(path_info[1])

    return A


@timer
def source_vector_0(space, f):
    qf = space.integrator
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = space.cellmeasure
    mesh = space.mesh
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    p = np.einsum('qj, cjk->qck', bcs, node[cell])
    fval = f(p) # (NQ, NC)
    phi = space.basis(bcs) # (NQ, 1, ldof)
    bb = np.einsum('q, qc, qci, c->ci', ws, fval, phi, cm, optimize=True)


@timer 
def integral_0(sapce, f):
    qf = space.integrator
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = space.cellmeasure
    mesh = space.mesh
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    p = np.einsum('qj, cjk->qck', bcs, node[cell])
    fval = f(p) # (NQ, NC)
    val = np.einsum('q, qc, c->c', ws, fval, cm, optimize=True)

@timer
def source_vector_1(space, f):
    qf = space.integrator
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = space.cellmeasure
    mesh = space.mesh
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    p = np.einsum('qj, cjk->cqk', bcs, node[cell])
    fval = f(p) # (NC, NQ)
    phi = space.basis(bcs) # (NQ, 1, ldof)
    bb = np.einsum('q, cq, qci, c->ci', ws, fval, phi, cm, optimize=True)


@timer 
def integral_1(sapce, f):
    qf = space.integrator
    bcs, ws = qf.get_quadrature_points_and_weights()
    cm = space.cellmeasure
    mesh = space.mesh
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    p = np.einsum('qj, cjk->cqk', bcs, node[cell])
    fval = f(p) # (NC, NQ)
    val = np.einsum('q, cq, c->c', ws, fval, cm, optimize=True)

n = 20 
p = 3
box = [0, 1]*3

mesh= MF.boxmesh3d(box, nx=n, ny=n, nz=n, meshtype='tet')
space = LagrangeFiniteElementSpace(mesh, p=p)


M = mass_matrix(space)
A = stiff_matrix(space)



cell2dof = space.cell_to_dof()
I = np.broadcast_to(cell2dof[:, :, None], shape=M.shape) 
J = np.broadcast_to(cell2dof[:, None, :], shape=M.shape)
gdof = space.number_of_global_dofs()
M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))

source_vector_0(space, f)
source_vector_1(space, f)
integral_0(space, f)
integral_1(space, f)




