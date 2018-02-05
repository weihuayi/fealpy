import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye



def stiff_matrix(V, qf, area, cfun=None, barycenter=True):
    mesh = V.mesh
    gdof = V.number_of_global_dofs()
    ldof = V.number_of_local_dofs()
    cell2dof = V.dof.cell2dof

    bcs, ws = qf.quadpts, qf.weights
    gphi = V.grad_basis(bcs)
    A = np.einsum('i, ijkm, ijpm->jkp', ws, gphi, gphi)
    A *= area.reshape(-1, 1, 1)
    I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
    J = I.swapaxes(-1, -2)
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
    return A

def mass_matrix(V, qf, area, cfun=None, barycenter=True):
    mesh = V.mesh
    gdof = V.number_of_global_dofs()
    ldof = V.number_of_local_dofs()
    cell2dof = V.dof.cell2dof

    bcs, ws = qf.quadpts, qf.weights
    phi = V.basis(bcs)
    if cfun is None:
        A = np.einsum('m, mj, mk, i->ijk', ws, phi, phi, area)
    else:
        if barycenter is True:
            val = cfun(bcs)
        else:
            pp = mesh.bc_to_point(bcs)
            val = cfun(pp)
        A = np.einsum('m, mi, mj, mk, i->ijk', ws, val, phi, phi, area)

    I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
    J = I.swapaxes(-1, -2)
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
    return A

def source_vector(f, V, qf, area):
    mesh = V.mesh
    
    bcs, ws = qf.quadpts, qf.weights
    pp = mesh.bc_to_point(bcs)
    fval = f(pp)
    phi = V.basis(bcs)
    bb = np.einsum('i, ij, ik->kj', ws, phi,fval)

    bb *= area.reshape(-1, 1)
    gdof = V.number_of_global_dofs()
    b = np.bincount(V.dof.cell2dof.flat, weights=bb.flat, minlength=gdof)
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
