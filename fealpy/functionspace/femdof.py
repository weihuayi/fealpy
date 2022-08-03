import numpy as np
import operator as op
from functools import reduce

def multi_index_matrix0d(p):
    multiIndex = 1
    return multiIndex 

def multi_index_matrix1d(p):
    ldof = p+1
    multiIndex = np.zeros((ldof, 2), dtype=np.int_)
    multiIndex[:, 0] = np.arange(p, -1, -1)
    multiIndex[:, 1] = p - multiIndex[:, 0]
    return multiIndex

def multi_index_matrix2d(p):
    ldof = (p+1)*(p+2)//2
    idx = np.arange(0, ldof)
    idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
    multiIndex = np.zeros((ldof, 3), dtype=np.int_)
    multiIndex[:,2] = idx - idx0*(idx0 + 1)/2
    multiIndex[:,1] = idx0 - multiIndex[:,2]
    multiIndex[:,0] = p - multiIndex[:, 1] - multiIndex[:, 2]
    return multiIndex

def multi_index_matrix3d(p):
    ldof = (p+1)*(p+2)*(p+3)//6
    idx = np.arange(1, ldof)
    idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3)
    idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
    idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
    idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
    multiIndex = np.zeros((ldof, 4), dtype=np.int_)
    multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
    multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
    multiIndex[1:, 1] = idx0 - idx2
    multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
    return multiIndex

multi_index_matrix = [multi_index_matrix0d, multi_index_matrix1d, multi_index_matrix2d, multi_index_matrix3d]


class CPLFEMDof1d():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = multi_index_matrix1d(p)
        self.cell2dof = self.cell_to_dof()

    def boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_node_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('node', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[index] = True
        return isBdDof

    def is_boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_node_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('node', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[index] = True
        return isBdDof

    def entity_to_dof(self, etype='cell', index=np.s_[:]):
        if etype in {'cell', 'face', 'edge', 1}:
            return self.cell_to_dof()[index]
        elif etype in {'node', 0}:
            NN = self.mesh.number_of_nodes()
            return np.arange(NN)[index]

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        if p == 1:
            return cell
        else:
            NN = mesh.number_of_nodes()
            NC = mesh.number_of_cells()
            ldof = self.number_of_local_dofs()
            cell2dof = np.zeros((NC, ldof), dtype=np.int)
            cell2dof[:, [0, -1]] = cell
            cell2dof[:, 1:-1] = NN + np.arange(NC*(p-1)).reshape(NC, p-1)
            return cell2dof

    def number_of_local_dofs(self, doftype='cell'):
        if doftype in {'cell', 'edge', 1}:
            return self.p + 1
        elif doftype in {'face', 'node', 0}:
            return 1

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        gdof = mesh.number_of_nodes()
        if p > 1:
            NC = mesh.number_of_cells()
            gdof += NC*(p-1)
        return gdof

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        node = mesh.node

        if p == 1:
            return node
        else:
            NN = mesh.number_of_nodes()
            gdof = self.number_of_global_dofs()
            shape = (gdof,) + node.shape[1:]
            ipoint = np.zeros(shape, dtype=np.float64)
            ipoint[:NN] = node
            NC = mesh.number_of_cells()
            cell = mesh.ds.cell
            w = np.zeros((p-1,2), dtype=np.float64)
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            GD = mesh.geo_dimension()
            ipoint[NN:NN+(p-1)*NC] = np.einsum('ij, kj...->ki...', w,
                    node[cell]).reshape(-1, GD)

            return ipoint

class CPLFEMDof2d():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = multi_index_matrix2d(p)
        self.cell2dof = self.cell_to_dof()

    def is_on_node_local_dof(self):
        p = self.p
        isNodeDof = np.sum(self.multiIndex == p, axis=-1) > 0
        return isNodeDof

    def is_on_edge_local_dof(self):
        return self.multiIndex == 0

    def boundary_dof(self, threshold=None):

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('edge', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        edge2dof = self.edge_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[edge2dof[index]] = True
        return isBdDof

    def is_boundary_dof(self, threshold=None):

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('edge', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        edge2dof = self.edge_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[edge2dof[index]] = True
        return isBdDof

    def entity_to_dof(self, etype='cell', index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.cell_to_dof()[index]
        elif etype in {'face', 'edge', 1}:
            return self.edge_to_dof()[index]
        elif etype in {'node', 0}:
            NN = self.mesh.number_of_nodes()
            return np.arange(NN)[index]

    def face_to_dof(self):
        return self.edge_to_dof()

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        NE= mesh.number_of_edges()
        NN = mesh.number_of_nodes()

        edge = mesh.entity('edge')
        edge2dof = np.zeros((NE, p+1), dtype=np.int_)
        edge2dof[:, [0, -1]] = edge
        if p > 1:
            edge2dof[:, 1:-1] = NN + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh

        cell = mesh.entity('cell')
        N = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()

        if p == 1:
            cell2dof = cell

        if p > 1:
            cell2dof = np.zeros((NC, ldof), dtype=np.int_)

            isEdgeDof = self.is_on_edge_local_dof()
            edge2dof = self.edge_to_dof()
            cell2edgeSign = mesh.ds.cell_to_edge_sign()
            cell2edge = mesh.ds.cell_to_edge()

            cell2dof[np.ix_(cell2edgeSign[:, 0], isEdgeDof[:, 0])] = \
                    edge2dof[cell2edge[cell2edgeSign[:, 0], [0]], :]
            cell2dof[np.ix_(~cell2edgeSign[:, 0], isEdgeDof[:,0])] = \
                    edge2dof[cell2edge[~cell2edgeSign[:, 0], [0]], -1::-1]

            cell2dof[np.ix_(cell2edgeSign[:, 1], isEdgeDof[:, 1])] = \
                    edge2dof[cell2edge[cell2edgeSign[:, 1], [1]], -1::-1]
            cell2dof[np.ix_(~cell2edgeSign[:, 1], isEdgeDof[:,1])] = \
                    edge2dof[cell2edge[~cell2edgeSign[:, 1], [1]], :]

            cell2dof[np.ix_(cell2edgeSign[:, 2], isEdgeDof[:, 2])] = \
                    edge2dof[cell2edge[cell2edgeSign[:, 2], [2]], :]
            cell2dof[np.ix_(~cell2edgeSign[:, 2], isEdgeDof[:,2])] = \
                    edge2dof[cell2edge[~cell2edgeSign[:, 2], [2]], -1::-1]
        if p > 2:
            base = N + (p-1)*NE
            isInCellDof = ~(isEdgeDof[:,0] | isEdgeDof[:,1] | isEdgeDof[:,2])
            idof = ldof - 3*p
            cell2dof[:, isInCellDof] = base + np.arange(NC*idof).reshape(NC, idof)

        return cell2dof

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')

        if p == 1:
            return node
        if p > 1:
            N = node.shape[0]
            dim = node.shape[-1]
            gdof = self.number_of_global_dofs()
            ipoint = np.zeros((gdof, dim), dtype=np.float64)
            ipoint[:N, :] = node
            NE = mesh.number_of_edges()
            edge = mesh.ds.edge
            w = np.zeros((p-1,2), dtype=np.float64)
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            ipoint[N:N+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', w,
                    node[edge,:]).reshape(-1, dim)
        if p > 2:
            isEdgeDof = self.is_on_edge_local_dof()
            isInCellDof = ~(isEdgeDof[:,0] | isEdgeDof[:,1] | isEdgeDof[:,2])
            w = self.multiIndex[isInCellDof, :]/p
            ipoint[N+(p-1)*NE:, :] = np.einsum('ij, kj...->ki...', w,
                    node[cell,:]).reshape(-1, dim)

        return ipoint

    def number_of_global_dofs(self):
        p = self.p
        N = self.mesh.number_of_nodes()
        gdof = N
        if p > 1:
            NE = self.mesh.number_of_edges()
            gdof += (p-1)*NE

        if p > 2:
            ldof = self.number_of_local_dofs()
            NC = self.mesh.number_of_cells()
            gdof += (ldof - 3*p)*NC
        return gdof

    def number_of_local_dofs(self, doftype='cell'):
        p = self.p
        if doftype in {'cell', 2}:
            return (p+1)*(p+2)//2 
        elif doftype in {'face', 'edge',  1}:
            return self.p + 1
        elif doftype in {'node', 0}:
            return 1

class CPLFEMDof3d():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = multi_index_matrix3d(p)
        self.multiIndex2d = multi_index_matrix2d(p)
        self.cell2dof = self.cell_to_dof()

    def is_on_node_local_dof(self):
        p = self.p
        isNodeDof = np.sum(self.multiIndex == p, axis=-1) == 1
        return isNodeDof

    def is_on_edge_local_dof(self):
        p =self.p
        ldof = self.number_of_local_dofs()
        localEdge = self.mesh.ds.localEdge
        isEdgeDof = np.zeros((ldof, 6), dtype=np.bool_)
        for i in range(6):
            isEdgeDof[:, i] = (self.multiIndex[:, localEdge[-(i+1), 0]] == 0) & (self.multiIndex[:, localEdge[-(i+1), 1]] == 0 )
        return isEdgeDof

    def is_on_face_local_dof(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        isFaceDof = (self.multiIndex == 0)
        return isFaceDof

    def entity_to_dof(self, etype='cell', index=np.s_[:]):
        if etype in {'cell', 3}:
            return self.cell_to_dof()[index]
        elif etype in {'face', 2}:
            return self.face_to_dof()[index]
        elif etype in {'edge', 1}:
            return self.edge_to_dof()[index]
        elif etype in {'node', 0}:
            NN = self.mesh.number_of_nodes()
            return np.arange(NN)[index]


    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        N = mesh.number_of_nodes()
        NE = mesh.number_of_edges()

        base = N
        edge = mesh.ds.edge
        edge2dof = np.zeros((NE, p+1), dtype=np.int_)
        edge2dof[:, [0, -1]] = edge
        if p > 1:
            edge2dof[:,1:-1] = base + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def face_to_dof(self):
        p = self.p
        fdof = (p+1)*(p+2)//2

        edgeIdx = np.zeros((2, p+1), dtype=np.int_)
        edgeIdx[0, :] = range(p+1)
        edgeIdx[1, :] = edgeIdx[0, -1::-1]

        mesh = self.mesh

        N = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()

        face = mesh.ds.face
        edge = mesh.ds.edge
        face2edge = mesh.ds.face_to_edge()

        edge2dof = self.edge_to_dof()

        face2dof = np.zeros((NF, fdof), dtype=np.int_)
        faceIdx = self.multiIndex2d
        isEdgeDof = (faceIdx == 0)

        fe = np.array([1, 0, 0])
        for i in range(3):
            I = np.ones(NF, dtype=np.int_)
            sign = (face[:, fe[i]] == edge[face2edge[:, i], 0])
            I[sign] = 0
            face2dof[:, isEdgeDof[:, i]] = edge2dof[face2edge[:, [i]], edgeIdx[I]]

        if p > 2:
            base = N + (p-1)*NE
            isInFaceDof = ~(isEdgeDof[:, 0] | isEdgeDof[:, 1] | isEdgeDof[:, 2])
            fidof = fdof - 3*p
            face2dof[:, isInFaceDof] = base + np.arange(NF*fidof).reshape(NF, fidof)

        return face2dof

    def boundary_dof(self, threshold=None):

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.face_to_dof()
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[face2dof[index]] = True
        return isBdDof

    def is_boundary_dof(self, threshold=None):

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.face_to_dof()
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[face2dof[index]] = True
        return isBdDof

    def cell_to_dof(self):
        p = self.p
        fdof = (p+1)*(p+2)//2
        ldof = self.number_of_local_dofs()

        localFace = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])

        mesh = self.mesh

        N = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        face = mesh.ds.face
        cell = mesh.ds.cell

        cell2face = mesh.ds.cell_to_face()

        cell2dof = np.zeros((NC, ldof), dtype=np.int_)

        face2dof = self.face_to_dof()
        isFaceDof = self.is_on_face_local_dof()
        faceIdx = self.multiIndex2d.T

        for i in range(4):
            fi = face[cell2face[:, i]]
            fj = cell[:, localFace[i]]
            idxj = np.argsort(fj, axis=1)
            idxjr = np.argsort(idxj, axis=1)
            idxi = np.argsort(fi, axis=1)
            idx = idxi[np.arange(NC).reshape(-1, 1), idxjr]
            isCase0 = (np.sum(idx == np.array([1, 2, 0]), axis=1) == 3)
            isCase1 = (np.sum(idx == np.array([2, 0, 1]), axis=1) == 3)
            idx[isCase0, :] = [2, 0, 1]
            idx[isCase1, :] = [1, 2, 0]
            k = faceIdx[idx[:, 1], :] + faceIdx[idx[:, 2], :]
            a = k*(k+1)//2 + faceIdx[idx[:, 2], :]
            cell2dof[:, isFaceDof[:, i]] = face2dof[cell2face[:, [i]], a]

        if p > 3:
            base = N + (p-1)*NE + (fdof - 3*p)*NF
            idof = ldof - 4 - 6*(p - 1) - 4*(fdof - 3*p)
            isInCellDof = ~(isFaceDof[:, 0] | isFaceDof[:, 1] | isFaceDof[:, 2] | isFaceDof[:, 3])
            cell2dof[:, isInCellDof] = base + np.arange(NC*idof).reshape(NC, idof)

        return cell2dof

    def cell_to_dof_1(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')

        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        cell2dof = np.zeros((NC, ldof), dtype=np.int_)

        idx = np.array([
            0,
            ldof - (p+1)*(p+2)//2 - 1,
            ldof - p -1,
            ldof - 1], dtype=np.int_)

        cell2dof[:, idx] = cell

        if p == 1:
            return cell2dof
        if p == 2:
            cell2edge = mesh.ds.cell_to_edge()
            idx = np.array([1, 2, 3, 5, 6, 8], dtype=np.int_)
            cel2dof[:, idx] = cell2edge + NN
            return cell2dof
        else:
            w = self.multiIndex

            flag = (w != 0)
            isCellIDof = (flag.sum(axis=-1) == 4)
            isNodeDof = (flag.sum(axis=-1) == 1)
            isNewBdDof = ~(isCellIDof | isNodeDof)

            nd = isNewBdDof.sum()
            ps = np.einsum('im, km->ikm', cell + NN + NC, w[isNewBdDof])
            ps.sort()
            _, i0, j = np.unique(
                    ps.reshape(-1, 4),
                    return_index=True,
                    return_inverse=True,
                    axis=0)
            cell2dof[:, isNewBdDof] = j.reshape(-1, nd) + NN

            NB = len(i0)
            nd = isCellIDof.sum()
            if nd > 0:
                cell2dof[:, isCellIDof] = NB + NN + nd*np.arange(NC).reshape(-1, 1) \
                        + np.arange(nd)
            return cell2dof

    def cell_to_dof_2(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')

        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        cell2dof = np.zeros((NC, ldof), dtype=np.int_)

        idx = np.array([
            0,
            ldof - (p+1)*(p+2)//2 - 1,
            ldof - p -1,
            ldof - 1], dtype=np.int_)

        cell2dof[:, idx] = cell

        if p == 1:
            return cell2dof
        if p == 2:
            cell2edge = mesh.ds.cell_to_edge()
            idx = np.array([1, 2, 3, 5, 6, 8], dtype=np.int_)
            cel2dof[:, idx] = cell2edge + NN
            return cell2dof
        else:
            w = self.multiIndex
            flag = (w != 0)
            isCellIDof = (flag.sum(axis=-1) == 4)
            nd = isCellIDof.sum()
            if nd > 0:
                cell2dof[:, isCellIDof] = (
                        NB + NN +
                        nd*np.arange(NC).reshape(-1, 1) + np.arange(nd)
                    )

            isNodeDof = (flag.sum(axis=-1) == 1)
            isNewBdDof = ~(isCellIDof | isNodeDof)
            # 边内部自由度编码
            m1 = multi_index_matrix(p, 1)
            edge = mesh.entity('edge')
            # 面内部自由度编码 
            m2 = multi_index_matrix(p, 2)
            face = mesh.entity('face')
            # 单元内部自由编码
            m3 = multi_index_matrix(p, 3)
            pass


    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        N = mesh.number_of_nodes()
        gdof = N

        if p > 1:
            NE = mesh.number_of_edges()
            edof = p - 1
            gdof += edof*NE

        if p > 2:
            NF = mesh.number_of_faces()
            fdof = (p+1)*(p+2)//2 - 3*p
            gdof += fdof*NF

        if p > 3:
            NC = mesh.number_of_cells()
            ldof = self.number_of_local_dofs()
            cdof = ldof - 6*edof - 4*fdof - 4
            gdof += cdof*NC

        return gdof


    def number_of_local_dofs(self, doftype='cell'):
        p = self.p
        if doftype in {'cell', 3}:
            return (p+1)*(p+2)*(p+3)//6
        elif doftype in {'face', 2}:
            return (p+1)*(p+2)//2
        elif doftype in {'edge', 1}:
            return p + 1
        elif doftype in {'node', 0}:
            return 1

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        node = mesh.node

        if p == 1:
            return node

        N = node.shape[0]
        dim = node.shape[1]
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        ipoint = np.zeros((gdof, dim), dtype=np.float64)
        ipoint[:N, :] = node
        if p > 1:
            NE = mesh.number_of_edges()
            edge = mesh.ds.edge
            w = np.zeros((p-1,2), dtype=np.float64)
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            ipoint[N:N+(p-1)*NE, :] = np.einsum('ij, kj...->ki...', w, node[edge,:]).reshape(-1, dim) 
        if p > 2:
            NF = mesh.number_of_faces()
            fidof = (p+1)*(p+2)//2 - 3*p
            face = mesh.ds.face
            isEdgeDof = (self.multiIndex2d == 0)
            isInFaceDof = ~(isEdgeDof[:, 0] | isEdgeDof[:, 1] | isEdgeDof[:, 2])
            w = self.multiIndex2d[isInFaceDof, :]/p
            ipoint[N+(p-1)*NE:N+(p-1)*NE+fidof*NF, :] = np.einsum('ij, kj...->ki...', w, node[face,:]).reshape(-1, dim)

        if p > 3:
            isFaceDof = self.is_on_face_local_dof()
            isInCellDof = ~(isFaceDof[:,0] | isFaceDof[:,1] | isFaceDof[:,2] | isFaceDof[:, 3])
            w = self.multiIndex[isInCellDof, :]/p
            ipoint[N+(p-1)*NE+fidof*NF:, :] = np.einsum('ij, kj...->ki...', w,
                    node[cell,:]).reshape(-1, dim)
        return ipoint


class DPLFEMDof():
    """
    间断单元自由度管理基类.
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = self.multi_index_matrix()
        self.cell2dof = self.cell_to_dof()


    def cell_to_dof(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        cell2dof = np.arange(NC*ldof).reshape(NC, ldof)
        return cell2dof

    def number_of_global_dofs(self):
        NC = self.mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        gdof = ldof*NC
        return gdof

    def number_of_local_dofs(self):
        p = self.p
        TD = self.mesh.top_dimension()
        numer = reduce(op.mul, range(p + TD, p, -1))
        denom = reduce(op.mul, range(1, TD + 1))
        return numer//denom

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        GD = mesh.geo_dimension()

        if p == 0:
            return mesh.entity_barycenter('cell')

        if p == 1:
            return node[cell].reshape(-1, GD)

        w = self.multiIndex/p
        ipoint = np.einsum('ij, kj...->ki...', w, node[cell]).reshape(-1, GD)
        return ipoint


class DPLFEMDof1d(DPLFEMDof):
    """
    区间间断单元自由度管理类.
    """
    def __init__(self, mesh, p):
        super(DPLFEMDof1d, self).__init__(mesh, p)

    def entity_to_dof(self, etype='cell', index=np.s_[:]):
        if etype in {'cell', 'face', 'edge', 1}:
            return self.cell_to_dof()[index]
        elif etype in {'node', 0}:
            return None # there is no dof on nodes 

    def multi_index_matrix(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        multiIndex = np.zeros((ldof, 2), dtype=np.int_)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex


class DPLFEMDof2d(DPLFEMDof):
    """
    三角形间断单元自由度管理类.
    """
    def __init__(self, mesh, p):
        super(DPLFEMDof2d, self).__init__(mesh, p)

    def entity_to_dof(self, etype='cell', index=np.s_[:]):
        if etype in {'cell',  2}:
            return self.cell_to_dof()[index]
        elif etype in {'face', 'edge', 1}:
            return None # there is no dof on nodes
        elif etype in {'node', 0}:
            return None # there is no dof on nodes 

    def multi_index_matrix(self):
        p = self.p
        if p == 0:
            return np.array([[0, 0, 0]], dtype=np.int_)
        ldof = self.number_of_local_dofs()
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 3), dtype=np.int_)
        multiIndex[:, 2] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 1] = idx0 - multiIndex[:, 2]
        multiIndex[:, 0] = p - multiIndex[:, 1] - multiIndex[:, 2]
        return multiIndex


class DPLFEMDof3d(DPLFEMDof):
    """
    四面体间断单元自由度管理类.
    """
    def __init__(self, mesh, p):
        super(DPLFEMDof3d, self).__init__(mesh, p)

    def entity_to_dof(self, etype='cell', index=np.s_[:]):
        if etype in {'cell', 3}:
            return self.cell_to_dof()[index]
        elif etype in {'face', 2}:
            return None # there is no dof on nodes
        elif etype in {'face', 1}:
            return None # there is no dof on nodes
        elif etype in {'node', 0}:
            return None # there is no dof on nodes 

    def multi_index_matrix(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        idx = np.arange(1, ldof)
        idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3)
        idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4)# a+b+c
        idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
        idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2)# b+c
        multiIndex = np.zeros((ldof, 4), dtype=np.int_)
        multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
        multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
        multiIndex[1:, 1] = idx0 - idx2
        multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
        return multiIndex



class CPPFEMDof3d():
    """
    三棱柱连续单元自由度管理类.
    """
    def __init__(self, mesh, p=1):
        self.mesh = mesh
        self.p = p
        self.cell2dof = self.cell_to_dof()
        self.dpoints = self.interpolation_points()

    def multi_index_matrix(self, TD):
        """
        一维和二维的多重指标.
        """
        p = self.p
        if TD == 1:
            multiIndex = multi_index_matrix1d(p)
            return multiIndex
        elif TD == 2:
            multiIndex = multi_index_matrix2d(p)
            return multiIndex

    def local_face_to_dof(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        idof = p+1
        tdof = (p+1)*(p+2)//2
        qdof = (p+1)*(p+1)


        idx = np.r_['0', [0, 3], 2*np.ones(p-1, dtype=np.int_)]
        f0 = np.repeat(np.cumsum(np.cumsum(idx)), range(1, p+2)) - np.arange(tdof)
        f1 = np.arange(tdof*p, ldof)
        f2 = np.repeat(
                np.arange(tdof - p - 1, ldof - p, tdof).reshape(1, -1)
                , p+1, 0) + np.arange(0, idof).reshape(-1, 1)
        idx = np.arange(1, p+2)
        idx[0] = 0
        f3 = np.repeat(
                np.arange(0, ldof-tdof+1, tdof).reshape(1, -1),
                p+1, 0) + np.cumsum(idx)[-1::-1].reshape(-1, 1)
        idx = np.arange(0, idof)
        f4 = np.repeat(
                np.arange(0, ldof-tdof+1, tdof).reshape(1, -1),
                p+1, 0) + np.cumsum(idx).reshape(-1, 1)
        localFace2dof = [f0.reshape(-1), f1.reshape(-1), f2.reshape(-1), f3.reshape(-1),
                f4.reshape(-1)]
        return localFace2dof


    def number_of_local_dofs(self):
        """
        每个单元上的自由度的个数.
        """
        p = self.p
        return (p+1)*(p+1)*(p+2)//2

    def number_of_global_dofs(self):
        """
        全部自由度的个数.
        """

        p = self.p
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NTF = mesh.number_of_tri_faces()
        NQF = mesh.number_of_quad_faces()
        NC = mesh.number_of_cells()
        gdof = NN

        if p > 1:
            gdof += NE*(p-1) + NQF*(p-1)*(p-1)

        if p > 2:
            tfdof = (p+1)*(p+2)//2 - 3*p
            gdof += NTF*tfdof
            gdof += NC*tfdof*(p-1)

        return gdof

    def boundary_dof(self, threshold=None):

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        face2dof = self.face_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[np.concatenate(face2dof[index])] = True
        return isBdDof

    def face_to_dof(self):
        localFace2dof = self.local_face_to_dof()
        face2cell = self.mesh.ds.face_to_cell()
        NF = self.mesh.number_of_faces()
        cell2dof = self.cell2dof
        f = lambda i: cell2dof[face2cell[i, 0], localFace2dof[face2cell[i, 2]]]
        face2dof = np.array(list(map(f, range(NF))), dtype=np.ndarray)
        return face2dof

    def edge_to_dof(self):
        pass

    def cell_to_dof(self):
        """
        每个单元对应的全局自由度的编号.
        """
        p = self.p
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        cell = mesh.entity('cell')
        if p == 1:
            return cell
        else:
            w0 = self.multi_index_matrix(1)
            w1 = self.multi_index_matrix(2)
            w = np.einsum('ij, km->ikjm', w0, w1)
            ldof0 = len(w0)
            ldof1 = len(w1)

            idx = cell.reshape(NC, 2, 3) + NN + NC
            # w: (ldof1, ldof2, 2, 3)
            # idx: (NC, 2, 3)
            ps = np.einsum('ijk, mnjk->imnjk', idx, w).reshape(NC, ldof0, ldof1, 6)
            ps.sort()
            t, self.i0, j = np.unique(
                    ps.reshape(-1, 6),
                    return_index=True,
                    return_inverse=True,
                    axis=0)
            return j.reshape(-1, ldof0*ldof1)

    def cell_to_dof_1(self):
        """
        每个单元对应的全局自由度的编号.
        """
        p = self.p
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        cell = mesh.entity('cell')
        if p == 1:
            return cell
        else:
            ldof = self.number_of_local_dofs()
            w1 = self.multi_index_matrix(1)
            w2 = self.multi_index_matrix(2)
            w3 = np.einsum('ij, km->ijkm', w1, w2)

            w = np.zeros((ldof, 6), dtype=np.int8)
            w[:, 0:3] = w3[:, 0, :, :].reshape(-1, 3)
            w[:, 3:] = w3[:, 1, :, :].reshape(-1, 3)

            ps = np.einsum('im, km->ikm', cell + (NN + NC), w)
            ps.sort()
            t, self.i0, j = np.unique(
                    ps.reshape(-1, 6),
                    return_index=True,
                    return_inverse=True,
                    axis=0)
            return j.reshape(-1, ldof)

    def cell_to_dof_2(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')

        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        cell2dof = np.zeros((NC, ldof), dtype=np.int_)
        idx = np.array([
            0,
            p*(p+1)//2,
            (p+1)*(p+2)//2-1,
            ldof - (p+1)*(p+2)//2,
            ldof - p - 1,
            ldof - 1], dtype=np.int_)
        cell2dof[:, idx] = cell

        if p == 1:
            return cell2dof
        else:
            w1 = self.multi_index_matrix(1)
            w2 = self.multi_index_matrix(2)
            w3 = np.einsum('ij, km->ijkm', w1, w2)

            w = np.zeros((ldof, 6), dtype=np.int8)
            w[:, 0:3] = w3[:, 0, :, :].reshape(-1, 3)
            w[:, 3:] = w3[:, 1, :, :].reshape(-1, 3)

            flag = (w != 0)
            isCellIDof = (flag.sum(axis=-1) == 6)
            isNodeDof = (flag.sum(axis=-1) == 1)
            isNewBdDof = ~(isCellIDof | isNodeDof)

            nd = isNewBdDof.sum()
            ps = np.einsum('im, km->ikm', cell + NN + NC, w[isNewBdDof])
            ps.sort()
            _, i0, j = np.unique(
                    ps.reshape(-1, 6),
                    return_index=True,
                    return_inverse=True,
                    axis=0)
            cell2dof[:, isNewBdDof] = j.reshape(-1, nd) + NN

            NB = len(i0)
            nd = isCellIDof.sum()
            if nd > 0:
                cell2dof[:, isCellIDof] = NB + NN + nd*np.arange(NC).reshape(-1, 1) \
                        + np.arange(nd)
            return cell2dof

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')

        if p == 1:
            return node

        GD = mesh.geo_dimension()

        ldof = self.number_of_local_dofs()
        w1 = self.multi_index_matrix(1)/p
        w2 = self.multi_index_matrix(2)/p
        w = np.einsum('ij, km->ikjm', w1, w2)

        ldof1 = len(w1)
        ldof2 = len(w2)
        NC = mesh.number_of_cells()
        idx = cell.reshape(NC, 2, 3)
        # node[idx]: (NC, 2, 3, GD)
        # w: (ldof1, ldof2, 2, 3)
        # ps: (NC, ldof1, ldof2, GD)
        ps = np.einsum('ijkm, nljk->inlm', node[idx], w).reshape(-1, GD)
        ipoint = ps[self.i0]
        return ipoint

    def interpolation_points_1(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')

        if p == 1:
            return node

        GD = mesh.geo_dimension()

        ldof = self.number_of_local_dofs()
        w1 = self.multi_index_matrix(1)/p
        w2 = self.multi_index_matrix(2)/p
        w3 = np.einsum('ij, km->ijkm', w1, w2)
        w = np.zeros((ldof, 6), dtype=np.float64)
        w[:, 0:3] = w3[:, 0, :, :].reshape(-1, 3)
        w[:, 3:] = w3[:, 1, :, :].reshape(-1, 3)
        ps = np.einsum('km, imd->ikd', w, node[cell]).reshape(-1, GD)
        ipoint = ps[self.i0]
        return ipoint
