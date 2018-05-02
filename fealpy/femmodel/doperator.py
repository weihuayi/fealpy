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
    return A, B

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

    localEdge = mesh.ds.local_edge()
    cellIdx = edge2cell[isBdEdge, [0]]
    localIdx = edge2cell[isBdEdge, 2]
    val0 = 0.5*h*n[:, 0]*gradphi[cellIdx, localEdge[localIdx], 0]  
    val0 = np.repeat(val0, 2, axis=0).reshape(-1, 2, 2)
    P0 = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))

    val0 = 0.5*h*n[:, 0]*gradphi[cellIdx, localEdge[localIdx], 1]  
    val0 = np.repeat(val0, 2, axis=0).reshape(-1, 2, 2)
    Q0 = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))

    val0 = 0.5*h*n[:, 1]*gradphi[cellIdx, localEdge[localIdx], 0]  
    val0 = np.repeat(val0, 2, axis=0).reshape(-1, 2, 2)
    P1 = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))

    val0 = 0.5*h*n[:, 1]*gradphi[cellIdx, localEdge[localIdx], 1]  
    val0 = np.repeat(val0, 2, axis=0).reshape(-1, 2, 2)
    Q1 = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(NN, NN))

    M0 = A.transpose()@P0@A + A.transpose()@Q0@B + B.transpose()@P1@A + B.transpose()@Q1@B

    M -= (M0 + M0.transpose())
    return  M 
