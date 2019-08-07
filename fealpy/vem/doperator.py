import numpy as np
from ..quadrature import GaussLobattoQuadrature, GaussLegendreQuadrature
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye
from numpy.linalg import inv


class BasicMatrix():
    def __init__(self, V, area):
        self.area = area
        self.H = matrix_H(V)
        self.D = matrix_D(V, self.H)
        self.B = matrix_B(V)
        self.C = matrix_C(V, self.B, self.D, self.H, self.area)
        self.G = matrix_G(V, self.B, self.D)

        self.PI0 = matrix_PI_0(V, self.H, self.C)
        self.PI1 = matrix_PI_1(V, self.G, self.B)

def basic_matrix(V, area):
    return BasicMatrix(V, area)

def stiff_matrix(V, area, cfun=None, mat=None):

    def f(x):
        x[0, :] = 0
        return x

    p = V.p
    if mat is None:
        pass
    else:
        G = mat.G
        PI1 = mat.PI1
        D = mat.D

    cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
    NC = len(cell2dofLocation) - 1
    cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
    DD = np.vsplit(D, cell2dofLocation[1:-1])

    if p == 1:
        tG = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
        if cfun is None:
            f1 = lambda x: x[1].T@tG@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
            K = list(map(f1, zip(DD, PI1)))
        else:
            barycenter = V.smspace.barycenter
            k = cfun(barycenter)
            f1 = lambda x: (x[1].T@tG@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1]))*x[2]
            K = list(map(f1, zip(DD, PI1, k)))
    else:
        tG = list(map(f, G))
        if cfun is None:
            f1 = lambda x: x[1].T@x[2]@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
            K = list(map(f1, zip(DD, PI1, tG)))
        else:
            barycenter = V.smspace.barycenter
            k = cfun(barycenter)
            f1 = lambda x: (x[1].T@x[2]@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1]))*x[3]
            K = list(map(f1, zip(DD, PI1, tG, k)))

    f2 = lambda x: np.repeat(x, x.shape[0])
    f3 = lambda x: np.tile(x, x.shape[0])
    f4 = lambda x: x.flatten()

    I = np.concatenate(list(map(f2, cd)))
    J = np.concatenate(list(map(f3, cd)))
    val = np.concatenate(list(map(f4, K)))
    gdof = V.number_of_global_dofs()
    A = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float)
    return A

def mass_matrix(V, area, cfun=None, mat=None):
    p = V.p
    if mat is None:
        pass
    else:
        PI0 = mat.PI0
        D = mat.D
        H = mat.H
        C = mat.C

    cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
    NC = len(cell2dofLocation) - 1
    cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
    DD = np.vsplit(D, cell2dofLocation[1:-1])

    f1 = lambda x: x[0]@x[1]
    PIS = list(map(f1, zip(DD, PI0)))

    f1 = lambda x: x[0].T@x[1]@x[0] + x[3]*(np.eye(x[2].shape[1]) - x[2]).T@(np.eye(x[2].shape[1]) - x[2])
    K = list(map(f1, zip(PI0, H, PIS, area)))

    f2 = lambda x: np.repeat(x, x.shape[0])
    f3 = lambda x: np.tile(x, x.shape[0])
    f4 = lambda x: x.flatten()

    I = np.concatenate(list(map(f2, cd)))
    J = np.concatenate(list(map(f3, cd)))
    val = np.concatenate(list(map(f4, K)))
    gdof = V.number_of_global_dofs()
    M = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float)
    return M

def cross_mass_matrix(integral, wh, vemspace, area, PI0):
    p = vemspace.p
    phi = vemspace.smspace.basis
    def u(x, cellidx):
        val = phi(x, cellidx=cellidx)
        wval = wh(x, cellidx=cellidx)
        return np.einsum('ij, ijm, ijn->ijmn', wval, val, val)
    H = integral(u, celltype=True)

    cell2dof, cell2dofLocation = vemspace.dof.cell2dof, vemspace.dof.cell2dofLocation
    NC = len(cell2dofLocation) - 1
    cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])

    f1 = lambda x: x[0].T@x[1]@x[0]
    K = list(map(f1, zip(PI0, H)))

    f2 = lambda x: np.repeat(x, x.shape[0])
    f3 = lambda x: np.tile(x, x.shape[0])
    f4 = lambda x: x.flatten()

    I = np.concatenate(list(map(f2, cd)))
    J = np.concatenate(list(map(f3, cd)))
    val = np.concatenate(list(map(f4, K)))
    gdof = vemspace.number_of_global_dofs()
    M = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float)
    return M

def source_vector(integral, f, vemspace, PI0):
    phi = vemspace.smspace.basis
    def u(x, cellidx):
        return np.einsum('ij, ijm->ijm', f(x), phi(x, cellidx=cellidx))
    bb = integral(u, celltype=True)
    g = lambda x: x[0].T@x[1]
    bb = np.concatenate(list(map(g, zip(PI0, bb))))
    gdof = vemspace.number_of_global_dofs()
    b = np.bincount(vemspace.dof.cell2dof, weights=bb, minlength=gdof)
    #print(b)
    return b

#def source_vector(f, V, area, vem=None):
#    cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
#    if V.p == 1:
#        mesh = V.mesh
#        ldof = V.number_of_local_dofs()
#        bb = np.zeros(ldof.sum(), dtype=np.float)
#        point = mesh.point
#        NV = mesh.number_of_vertices_of_cells()
#        F = f(point)
#        bb = F[cell2dof]/np.repeat(NV, NV)*np.repeat(area, NV)
#    else:
#        if vem is not None:
#            qf = vem.error.integrator  
#            bcs, ws = qf.quadpts, qf.weights
#            pp = vem.quadtree.bc_to_point(bcs)
#            val = f(pp)
#            phi = vem.V.smspace.basis(pp)
#            bb = np.einsum('i, ij, ijm->jm', ws, val, phi)
#            bb *= vem.area[:, np.newaxis]
#            g = lambda x: x[0].T@x[1]
#            bb = np.concatenate(list(map(g, zip(vem.PI0, bb))))
#        else:
#            raise ValueError('We need vem!')
#            
#    gdof = V.number_of_global_dofs()
#    b = np.bincount(cell2dof, weights=bb, minlength=gdof)
#    return b

def matrix_H(V):
    p = V.p
    mesh = V.mesh
    node = mesh.entity('node')

    edge = mesh.entity('edge')
    edge2cell = mesh.ds.edge_to_cell()

    isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

    NC = mesh.number_of_cells()

    qf = GaussLegendreQuadrature(p + 1)
    bcs, ws = qf.quadpts, qf.weights
    ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
    phi0 = V.smspace.basis(ps, cellidx=edge2cell[:, 0])
    phi1 = V.smspace.basis(ps[:, isInEdge, :], cellidx=edge2cell[isInEdge, 1])
    H0 = np.einsum('i, ijk, ijm->jkm', ws, phi0, phi0)
    H1 = np.einsum('i, ijk, ijm->jkm', ws, phi1, phi1)

    nm = mesh.edge_normal()
    b = node[edge[:, 0]] - V.smspace.barycenter[edge2cell[:, 0]]
    H0 = np.einsum('ij, ij, ikm->ikm', b, nm, H0)
    b = node[edge[isInEdge, 0]] - V.smspace.barycenter[edge2cell[isInEdge, 1]]
    H1 = np.einsum('ij, ij, ikm->ikm', b, -nm[isInEdge], H1)

    ldof = V.smspace.number_of_local_dofs()
    H = np.zeros((NC, ldof, ldof), dtype=np.float)
    np.add.at(H, edge2cell[:, 0], H0)
    np.add.at(H, edge2cell[isInEdge, 1], H1)

    multiIndex = V.smspace.dof.multiIndex
    q = np.sum(multiIndex, axis=1)
    H /= q + q.reshape(-1, 1) + 2
    return H

def matrix_H_test(V, vem=None):
    qf = vem.error.integrator
    bcs, ws = qf.quadpts, qf.weights
    pp = vem.quadtree.bc_to_point(bcs)
    phi = vem.V.smspace.basis(pp)
    H = np.einsum('i, ijk, ijm->jkm', ws, phi, phi)
    H *= vem.area[:, np.newaxis, np.newaxis]
    return H

def matrix_D(V, H):
    p = V.p
    smldof = V.smspace.number_of_local_dofs()
    mesh = V.mesh
    NV = mesh.number_of_vertices_of_cells()
    h = V.smspace.h
    node = mesh.node
    edge = mesh.ds.edge
    edge2cell = mesh.ds.edge2cell
    isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

    cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
    D = np.ones((len(cell2dof), smldof), dtype=np.float)

    if p == 1:
        bc = np.repeat(V.smspace.barycenter, NV, axis=0)
        D[:, 1:] = (node[mesh.ds.cell, :] - bc)/np.repeat(h, NV).reshape(-1, 1)
        return D

    qf = GaussLobattoQuadrature(p+1)
    bcs, ws = qf.quadpts, qf.weights
    ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
    phi0 = V.smspace.basis(ps[:-1], cellidx=edge2cell[:, 0])
    phi1 = V.smspace.basis(ps[p:0:-1, isInEdge, :], cellidx=edge2cell[isInEdge, 1])
    idx = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p + np.arange(p).reshape(-1, 1)
    D[idx, :] = phi0
    idx = cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p + np.arange(p).reshape(-1, 1)
    D[idx, :] = phi1
    if p > 1:
        area = V.smspace.area
        idof = (p-1)*p//2 # the number of dofs of scale polynomial space with degree p-2
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
    if p == 1:
        B[0, :] = 1/np.repeat(NV, NV)
        B[1:, :] = mesh.node_normal().T/np.repeat(h, NV).reshape(1, -1)
        return B
    else:
        idx = cell2dofLocation[0:-1] + NV*p
        B[0, idx] = 1
        idof = (p-1)*p//2
        start = 3
        r = np.arange(1, p+1)
        r = r[0:-1]*r[1:]
        for i in range(2, p+1):
            idx0 = np.arange(start, start+i-1)
            idx1 =  np.arange(start-2*i+1, start-i)
            idx1 = idx.reshape(-1, 1) + idx1
            B[idx0, idx1] -= r[i-2::-1]
            B[idx0+2, idx1] -= r[0:i-1]
            start += i+1
        node = mesh.node
        edge = mesh.ds.edge
        edge2cell = mesh.ds.edge2cell
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        qf = GaussLobattoQuadrature(p + 1)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        gphi0 = V.smspace.grad_basis(ps, cellidx=edge2cell[:, 0])
        gphi1 = V.smspace.grad_basis(ps[-1::-1, isInEdge, :], cellidx=edge2cell[isInEdge, 1])
        nm = mesh.edge_normal()

        val = np.einsum('ijmk, jk->jmi', gphi0, nm)
        val = np.einsum('i, jmi->jmi', ws, val)

        #TODO: vectorize
        NE = mesh.number_of_edges()
        for i in range(NE):
            idx0 = edge2cell[i, 0]
            idx1 = cell2dofLocation[idx0] + (edge2cell[i, 2]*p + np.arange(p+1))%(NV[idx0]*p)
            B[:, idx1] += val[i]

        val = np.einsum('ijmk, jk->jmi', gphi1, -nm[isInEdge])
        val = np.einsum('i, jmi->jmi', ws, val)

        #TODO: vectorize
        j = 0
        for i in range(NE):
            if isInEdge[i]:
                idx0 = edge2cell[i, 1]
                idx1 = cell2dofLocation[idx0] + (edge2cell[i, 3]*p + np.arange(p+1))%(NV[idx0]*p)
                B[:, idx1] += val[j]
                j += 1
        return B

def matrix_G(V, B, D):
    p = V.p
    if p == 1:
        G = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    else:
        cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
        BB = np.hsplit(B, cell2dofLocation[1:-1])
        DD = np.vsplit(D, cell2dofLocation[1:-1])
        g = lambda x: x[0]@x[1]
        G = list(map(g, zip(BB, DD)))
    return G

def matrix_G_test(V, vem=None):
    qf = vem.error.integrator
    bcs, ws = qf.quadpts, qf.weights
    pp = vem.quadtree.bc_to_point(bcs)
    gphi = vem.V.smspace.grad_basis(pp)
    G = np.einsum('i, ijkl, ijml->jkm', ws, gphi, gphi)
    G *= vem.area[:, np.newaxis, np.newaxis]
    G[:, 0, :] = vem.H[:, 0, :]/vem.area[:, np.newaxis]
    return G


def matrix_C(V, B, D, H, area):
    p = V.p

    smldof = V.smspace.number_of_local_dofs()
    idof = (p-1)*p//2

    mesh = V.mesh
    NV = mesh.number_of_vertices_of_cells()
    cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
    BB = np.hsplit(B, cell2dofLocation[1:-1])
    DD = np.vsplit(D, cell2dofLocation[1:-1])
    g = lambda x: x[0]@x[1]
    G = list(map(g, zip(BB, DD)))
    d = lambda x: x[0]@inv(x[1])@x[2]
    C = list(map(d, zip(H, G, BB)))
    if p == 1:
        return C
    else:
        l = lambda x: np.r_[
                '0',
                np.r_['1', np.zeros((idof, p*x[0])), x[1]*np.eye(idof)],
                x[2][idof:, :]]
        return list(map(l, zip(NV, area, C)))

def matrix_PI_0(V, H, C):
    cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
    pi0 = lambda x: inv(x[0])@x[1]
    return list(map(pi0, zip(H, C)))
        
def matrix_PI_1(V, G, B):
    p = V.p
    if p == 1:
        cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
        return np.hsplit(B, cell2dofLocation[1:-1])
    else:
        cell2dof, cell2dofLocation = V.dof.cell2dof, V.dof.cell2dofLocation
        BB = np.hsplit(B, cell2dofLocation[1:-1])
        g = lambda x: inv(x[0])@x[1]
        return list(map(g, zip(G, BB)))


