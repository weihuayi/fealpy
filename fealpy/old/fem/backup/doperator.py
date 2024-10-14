import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye
from timeit import default_timer as timer
from itertools import combinations

def stiff_matrix(space, qf, measure, cfun=None, barycenter=True):
    bcs, ws = qf.quadpts, qf.weights
    gphi = space.grad_basis(bcs)

    # Compute the element sitffness matrix
    A = np.einsum('i, ijkm, ijpm, j->jkp', ws, gphi, gphi, measure, optimize=True)
    cell2dof = space.cell_to_dof()
    ldof = space.number_of_local_dofs()
    I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
    J = I.swapaxes(-1, -2)
    gdof = space.number_of_global_dofs()

    # Construct the stiffness matrix
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
    return A

def stiff_matrix_1(space, qf, measure):
    bcs, ws = qf.quadpts, qf.weights
    gphi = space.grad_basis(bcs)
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    ldof = space.number_of_local_dofs()

    D = np.einsum('i, ijkm, ijkm, j->jk', ws, gphi, gphi, measure, optimize=True)
    A = coo_matrix((D.flat, (cell2dof.flat, cell2dof.flat)), shape=(gdof, gdof))
    for i, j in combinations(range(ldof), 2):
        D = np.einsum('i, ijm, ijm, j->j', ws, gphi[..., i, :], gphi[..., j, :], measure, optimize=True)
        A += coo_matrix((D, (cell2dof[:, i], cell2dof[:, j])), shape=(gdof, gdof))
        A += coo_matrix((D, (cell2dof[:, j], cell2dof[:, i])), shape=(gdof, gdof))

    return A.tocsr() 


def mass_matrix(space, qf, measure, cfun=None, barycenter=True):

    bcs, ws = qf.quadpts, qf.weights
    phi = space.basis(bcs)
    if cfun is None:
        A = np.einsum('m, mij, mik, i->ijk', ws, phi, phi, measure)
    else:
        if barycenter is True:
            val = cfun(bcs)
        else:
            pp = space.mesh.bc_to_point(bcs)
            val = cfun(pp)
        A = np.einsum('m, mi, mij, mik, i->ijk', ws, val, phi, phi, measure)

    cell2dof = space.cell_to_dof()
    ldof = space.number_of_local_dofs()
    I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
    J = I.swapaxes(-1, -2)

    gdof = space.number_of_global_dofs()
    A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
    return A

def source_vector(f, space, qf, measure, surface=None):
    bcs, ws = qf.quadpts, qf.weights
    pp = space.mesh.bc_to_point(bcs)
    if surface is not None:
        pp, _ = surface.project(pp)
    fval = f(pp)
    phi = space.basis(bcs)
    bb = np.einsum('i, ik, i..., k->k...', ws, fval, phi, measure)

    cell2dof = space.dof.cell2dof
    gdof = space.number_of_global_dofs()
    b = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)
    return b

def grad_recovery_matrix(space, rtype='simple'):
    mesh = space.mesh
    NC = mesh.number_of_cells() 
    NN = mesh.number_of_nodes() 
    cell = mesh.entity('cell')

    area = mesh.entity_measure('cell')
    gradphi = mesh.grad_lambda()

    if rtype is 'simple':
        D = spdiags(1.0/np.bincount(cell.flat), 0, NN, NN)
        I = np.einsum('k, ij->ijk', np.ones(3), cell)
        J = I.swapaxes(-1, -2)
        val = np.einsum('k, ij->ikj', np.ones(3), gradphi[:, :, 0])
        A = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
        val = np.einsum('k, ij->ikj', np.ones(3), gradphi[:, :, 1])
        B = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
    elif fem.rtype is 'harmonic':
        gphi = gradphi/area.reshape(-1, 1, 1)
        d = np.zeros(NN, dtype=np.float)
        np.add.at(d, cell, 1/area.reshape(-1, 1))
        D = spdiags(1/d, 0, N, N)
        I = np.einsum('k, ij->ijk', np.ones(3), cell)
        J = I.swapaxes(-1, -2)
        val = np.einsum('ij, k->ikj',  gphi[:, :, 0], np.ones(3))
        A = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
        val = np.einsum('ij, k->ikj',  gphi[:, :, 1], np.ones(3))
        B = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
    else:
        raise ValueError("I have not coded the method {}".format(rtype))
    return A, B, gradphi

def recovery_biharmonic_matirx(space, area, A, B, epsilon):
    mesh = space.mesh
    gradphi = mesh.grad_lambda() 
    NC = mesh.number_of_cells() 
    NN = mesh.number_of_nodes() 

    node = mesh.entity('node')
    edge = mesh.entity('edge')
    cell = mesh.entity('cell')

    edge2cell = mesh.ds.edge_to_cell()
    isBdEdge = (edge2cell[:,0]==edge2cell[:,1])
    bdEdge = edge[isBdEdge]
    
    # construct the unit outward normal on the boundary
    W = np.array([[0, -1], [1, 0]], dtype=np.int)
    n = (node[bdEdge[:,1],] - node[bdEdge[:,0],:])@W
    h = np.sqrt(np.sum(n**2, axis=1)) 
    n /= h.reshape(-1, 1)

    I = np.einsum('ij, k->ijk',  cell, np.ones(3))
    J = I.swapaxes(-1, -2)
    val = np.einsum('i, ij, ik->ijk', area, gradphi[:, :, 0], gradphi[:, :, 0])
    P = csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
    val = np.einsum('i, ij, ik->ijk', area, gradphi[:, :, 0], gradphi[:, :, 1])
    Q = csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
    val = np.einsum('i, ij, ik->ijk', area, gradphi[:, :, 1], gradphi[:, :, 1])
    S = csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))

    M = A.transpose()@P@A + A.transpose()@Q@B + B.transpose()@Q.transpose()@A+B.transpose()@S@B 
    M *= epsilon**2

    I = np.einsum('ij, k->ijk', bdEdge, np.ones(2))
    J = I.swapaxes(-1, -2)
    val = np.array([(1/3, 1/6), (1/6, 1/3)])
    val0 = np.einsum('i, jk->ijk', n[:, 0]*n[:, 0]/h, val)
    P = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))
    val0 = np.einsum('i, jk->ijk', n[:, 0]*n[:, 1]/h, val)
    Q = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))
    val0 = np.einsum('i, jk->ijk', n[:, 1]*n[:, 1]/h, val)
    S = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))

    M += A.transpose()@P@A + A.transpose()@Q@B + B.transpose()@Q@A + B.transpose()@S@B
    return  M 
