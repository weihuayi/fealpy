import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

def stiff_matrix(space, qf, measure, cfun=None, barycenter=True):
    bcs, ws = qf.quadpts, qf.weights
    gphi = space.grad_basis(bcs)
    A = np.einsum('i, ijkm, ijpm, j->jkp', ws, gphi, gphi, measure)
    
    cell2dof = space.cell_to_dof()
    ldof = space.number_of_local_dofs()
    I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
    J = I.swapaxes(-1, -2)

    gdof = space.number_of_global_dofs()
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
    return A

def mass_matrix(space, qf, measure, cfun=None, barycenter=True):

    bcs, ws = qf.quadpts, qf.weights
    phi = space.basis(bcs)
    if cfun is None:
        A = np.einsum('m, mj, mk, i->ijk', ws, phi, phi, measure)
    else:
        if barycenter is True:
            val = cfun(bcs)
        else:
            pp = space.mesh.bc_to_point(bcs)
            val = cfun(pp)
        A = np.einsum('m, mi, mj, mk, i->ijk', ws, val, phi, phi, measure)

    cell2dof = space.cell_to_dof()
    ldof = space.number_of_local_dofs()
    I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
    J = I.swapaxes(-1, -2)

    gdof = space.number_of_global_dofs()
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
    return A

def source_vector(f, space, qf, measure):
    
    bcs, ws = qf.quadpts, qf.weights
    pp = space.mesh.bc_to_point(bcs)
    fval = f(pp)
    phi = space.basis(bcs)
    bb = np.einsum('i, ik, ij, k->kj', ws, fval, phi, measure)

    cell2dof = space.dof.cell2dof
    gdof = space.number_of_global_dofs()
    b = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)
    return b

def grad_recovery_matrix(fem):
    V = fem.V
    mesh = V.mesh
    gradphi = mesh.grad_lambda() 

    NC = mesh.number_of_cells() 
    N = mesh.number_of_points() 
    cell = mesh.ds.cell

    if fem.rtype is 'simple':
        D = spdiags(1.0/np.bincount(cell.flat), 0, N, N)
        I = np.einsum('k, ij->ijk', np.ones(3), cell)
        J = I.swapaxes(-1, -2)
        val = np.einsum('k, ij->ikj', np.ones(3), gradphi[:, :, 0])
        A = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
        val = np.einsum('k, ij->ikj', np.ones(3), gradphi[:, :, 1])
        B = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
    elif fem.rtype is 'harmonic':
        area = fem.area
        gphi = gradphi/area.reshape(-1, 1, 1)
        d = np.zeros(N, dtype=np.float)
        np.add.at(d, cell, 1/area.reshape(-1, 1))
        D = spdiags(1/d, 0, N, N)
        I = np.einsum('k, ij->ijk', np.ones(3), cell)
        J = I.swapaxes(-1, -2)
        val = np.einsum('ij, k->ikj',  gphi[:, :, 0], np.ones(3))
        A = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
        val = np.einsum('ij, k->ikj',  gphi[:, :, 1], np.ones(3))
        B = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
    else:
        raise ValueError("I have not coded the method {}".format(fem.rtype))
    return A, B
