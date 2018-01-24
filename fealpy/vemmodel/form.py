import numpy as np
from ..quadrature import GaussLobattoQuadrature, GaussLegendreQuadrture 
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

def stiff_matrix(fem):
    V = fem.V
    p = V.p
    fem.H = matrix_H(V)
    fem.D = matrix_D(V, fem.H)
    fem.B = matrix_B(V)

    cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
    cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
    BB = np.hsplit(fem.B, cell2dofLocation[1:-1])
    DD = np.vsplit(fem.D, cell2dofLocation[1:-1])
            
    f1 = lambda x: (x[1].T@tG@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1]))*x[2]
    f2 = lambda x: np.repeat(x, x.shape[0]) 
    f3 = lambda x: np.tile(x, x.shape[0])
    f4 = lambda x: x.flatten()

    if p == 1:
        tG = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
    try:
        barycenter = V.smspace.barycenter 
        k = fem.model.diffusion_coefficient(barycenter)
    except  AttributeError:
        k = np.ones(NC) 

    K = list(map(f1, zip(DD, BB, k)))
    I = np.concatenate(list(map(f2, cd)))
    J = np.concatenate(list(map(f3, cd)))
    val = np.concatenate(list(map(f4, K)))
    gdof = V.number_of_global_dofs()
    A = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float)
    return A

def source_vector(fem):
    V = fem.V
    mesh = V.mesh
    model = fem.model

    ldof = V.number_of_local_dofs()
    bb = np.zeros(ldof.sum(), dtype=np.float)
    point = mesh.point
    NV = mesh.number_of_vertices_of_cells()
    F = model.source(point)
    area = V.smspace.area
    cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
    bb = F[cell2dof]/np.repeat(NV, NV)*np.repeat(area, NV)
    gdof = V.number_of_global_dofs()
    b = np.bincount(cell2dof, weights=bb, minlength=gdof)
    return b

def matrix_H(V):
    p = V.p
    mesh = V.mesh
    point = mesh.point

    edge = mesh.ds.edge
    edge2cell = mesh.ds.edge2cell

    isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

    NC = mesh.number_of_cells()

    qf = GaussLegendreQuadrture(p + 1)
    bcs, ws = qf.quadpts, qf.weights 
    ps = np.einsum('ij, kjm->ikm', bcs, point[edge])
    phi0 = V.smspace.basis(ps, cellidx=edge2cell[:, 0])
    phi1 = V.smspace.basis(ps[:, isInEdge, :], cellidx=edge2cell[isInEdge, 1])
    H0 = np.einsum('i, ijk, ijm->jkm', ws, phi0, phi0)
    H1 = np.einsum('i, ijk, ijm->jkm', ws, phi1, phi1) 

    nm = mesh.edge_normal()
    b = point[edge[:, 0]] - V.smspace.barycenter[edge2cell[:, 0]]
    H0 = np.einsum('ij, ij, ikm->ikm', b, nm, H0)
    b = point[edge[isInEdge, 0]] - V.smspace.barycenter[edge2cell[isInEdge, 1]]
    H1 = np.einsum('ij, ij, ikm->ikm', b, -nm[isInEdge], H1)

    ldof = V.smspace.number_of_local_dofs()
    H = np.zeros((NC, ldof, ldof), dtype=np.float)
    np.add.at(H, edge2cell[:, 0], H0)
    np.add.at(H, edge2cell[isInEdge, 1], H1)

    multiIndex = V.smspace.dof.multiIndex
    q = np.sum(multiIndex, axis=1)
    H /= q + q.reshape(-1, 1) + 2
    return H

def matrix_D(V, H):
    p = V.p
    smldof = V.smspace.number_of_local_dofs()
    mesh = V.mesh
    NV = mesh.number_of_vertices_of_cells()
    h = V.smspace.h 
    point = mesh.point
    edge = mesh.ds.edge
    edge2cell = mesh.ds.edge2cell
    isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

    cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation 
    D = np.zeros((len(cell2dof), smldof), dtype=np.float)

    if p == 1:
        bc = np.repeat(V.smspace.barycenter, NV, axis=0) 
        D[:, 1:] = (point[mesh.ds.cell, :] - bc)/np.repeat(h, NV).reshape(-1, 1)


    qf = GaussLobattoQuadrature(p + 1)
    bcs, ws = qf.quadpts, qf.weights 
    ps = np.einsum('ij, kjm->ikm', bcs[:-1], point[edge])
    phi0 = V.smspace.basis(ps, cellidx=edge2cell[:, 0])
    phi1 = V.smspace.basis(ps[-1::-1, isInEdge, :], cellidx=edge2cell[isInEdge, 1])
    idx = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p + np.arange(p).reshape(-1, 1)  
    D[idx, :] = phi0
    idx = cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p + np.arange(p).reshape(-1, 1)
    D[idx, :] = phi1
    if p > 1:
        area = V.smspace.area
        idof = int((p-1)*p/2) # the number of dofs of scale polynomial space with degree p-2
        idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-idof, 0)
        D[idx, :] = H[:, :idof, :]/area.reshape(-1, 1, 1)
    return D

def matrix_B(V):
    p = V.p
    smldof = V.smspace.number_of_local_dofs()
    mesh = V.mesh
    NV = mesh.number_of_vertices_of_cells()
    h = V.smspace.h 
    cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
    B = np.zeros((smldof, cell2dof.shape[0]), dtype=np.float) 
    if p==1:
        B[0, :] = 1/np.repeat(NV, NV)
        B[1:, :] = mesh.node_normal().T/np.repeat(h, NV).reshape(1, -1)
    else:
        idx = cell2dofLocation[0:-1] + NV*p 
        B[0, idx] = 1
        idof = (p-1)*p//2
        start = 3
        r = np.r_[1, np.arange(1, p+1)]
        r = np.cumprod(r)
        r = r[2:]/r[0:-2]
        for i in range(2, p+1):
            idx0 = np.arange(start, start+i-1)
            idx1 = np.arange(start-2*i+1, start-i)
            idx1 = idx.reshape(-1, 1) + idx1.reshape(1, -1)
            B[idx0, idx1] -= r[i-2::-1]
            B[idx0+2, idx1] -= r[0:i-1]
            start += i+1

        point = mesh.point
        edge = mesh.ds.edge
        edge2cell = mesh.ds.edge2cell
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        qf = GaussLobattoQuadrature(p + 1)
        bcs, ws = qf.quadpts, qf.weights 
        ps = np.einsum('ij, kjm->ikm', bcs, point[edge])
        gphi0 = V.smspace.grad_basis(ps, cellidx=edge2cell[:, 0])
        gphi1 = V.smspace.grad_basis(ps[-1::-1, isInEdge, :], cellidx=edge2cell[isInEdge, 1])
        nm = mesh.edge_normal()

        val = np.eisum('ijmk, jk->jmi', gphi0, nm)
        val = np.einsum('i, jmi->jmi', ws, val)

        #TODO: vectorize
        for i in range(NE):
            idx0 = edge2cell[i, 0] 
            idx1 = cell2dofLocation[idx0] + edge2cell[i, 2]*p + np.arange(p+1)
            B[:, idx1] += val[i]
    
        val = np.eisum('ijmk, jk->jmi', gphi1, -nm[isInEdge])
        val = np.einsum('i, jmi->jmi', ws, val)

        #TODO: vectorize
        j = 0
        for i in range(NE):
            if isInEdge[i]:
                idx0 = edge2cell[i, 1]
                idx1 = cell2dofLocation[idx0] + edge2cell[i, 3]*p + np.arange(p+1)
                B[:, idx1] += val[j]
                j += 1
    return B


