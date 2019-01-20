import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, bmat, eye
from .Mesh2d import Mesh2d, Mesh2dDataStructure
from ..quadrature import TriangleQuadrature


class TriangleMeshDataStructure(Mesh2dDataStructure):
    localEdge = np.array([(1, 2), (2, 0), (0, 1)])
    V = 3
    E = 3
    F = 1
    def __init__(self, N, cell):
        super(TriangleMeshDataStructure, self).__init__(N, cell) 

class TriangleMesh(Mesh2d):
    def __init__(self, node, cell):

        self.node = node
        N = node.shape[0]
        self.ds = TriangleMeshDataStructure(N, cell)
        if node.shape[1] == 2:
            self.meshtype = 'tri'
        elif node.shape[1] == 3:
            self.meshtype = 'stri'

        self.itype = cell.dtype
        self.ftype = node.dtype

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}

    def integrator(self, k):
        return TriangleQuadrature(k)

    def delete_cell(self, dflag):
        cell = self.entity('cell')
        cell = cell[~dflag]
        NN = self.number_of_nodes()
        self.ds.reinit(NN, cell)
        

    def circumcenter(self):
        node = self.node
        cell = self.ds.cell
        dim = self.geo_dimension()

        v0 = node[cell[:,2],:] - node[cell[:,1],:]
        v1 = node[cell[:,0],:] - node[cell[:,2],:]
        v2 = node[cell[:,1],:] - node[cell[:,0],:]
        nv = np.cross(v2, -v1)
        if dim == 2:
            area = nv/2.0 
            x2 = np.sum(node**2, axis=1, keepdims=True)
            w0 = x2[cell[:,2]] + x2[cell[:,1]]
            w1 = x2[cell[:,0]] + x2[cell[:,2]]
            w2 = x2[cell[:,1]] + x2[cell[:,0]]
            W = np.array([[0, -1],[1, 0]], dtype=self.ftype)
            fe0 = w0*v0@W 
            fe1 = w1*v1@W
            fe2 = w2*v2@W 
            c = 0.25*(fe0 + fe1 + fe2)/area.reshape(-1,1)
            R = np.sqrt(np.sum((c-node[cell[:,0], :])**2,axis=1))
        elif dim == 3:
            length = np.sqrt(np.sum(nv**2, axis=1))
            n = nv/length.reshape((-1, 1))
            l02 = np.sum(v1**2, axis=1, keepdims=True)
            l01 = np.sum(v2**2, axis=1, keepdims=True)
            d = 0.5*(l02*np.cross(n, v2) + l01*np.cross(-v1, n))/length.reshape(-1, 1)
            c = node[cell[:, 0]] + d
            R = np.sqrt(np.sum(d**2, axis=1))
        return c, R

    def angle(self):
        NC = self.number_of_cells()
        cell = self.ds.cell
        node = self.node
        localEdge = self.ds.local_edge()
        angle = np.zeros((NC, 3), dtype=self.ftype)
        for i,(j,k) in zip(range(3),localEdge):
            v0 = node[cell[:,j]] - node[cell[:,i]]
            v1 = node[cell[:,k]] - node[cell[:,i]]
            angle[:,i] = np.arccos(np.sum(v0*v1, axis=1)/np.sqrt(np.sum(v0**2, axis=1) * np.sum(v1**2, axis=1)))
        return angle

    def edge_swap(self):
        while True:
            # Construct necessary data structure
            edge2cell = self.ds.edge_to_cell()
            cell2edge = self.ds.cell_to_edge()

            # Find non-Delaunay edges
            angle = self.angle()
            asum = np.sum(angle[edge2cell[:, 0:2], edge2cell[:, 2:4]], axis=1)
            isNonDelaunayEdge = (asum > np.pi) & (edge2cell[:,0] != edge2cell[:,1])

            return isNonDelaunayEdge
            
            if np.sum(isNonDelaunayEdge) == 0:
                break
            # Find dependent set of swap edges
            isCheckCell = np.sum(isNonDelaunayEdge[cell2edge], axis=1) > 1
            if np.any(isCheckCell):
                ac = asum[cell2edge[isCheckCell, :]]
                isNonDelaunayEdge[cell2edge[isCheckCell, :]] = False
                I = np.argmax(ac, axis=1)
                isNonDelaunayEdge[cell2edge[isCheckCell, I]] = True

            if np.any(isNonDelaunayEdge):
                cell = self.ds.cell
                pnext = np.array([1, 2, 0])
                idx = edge2cell[isNonDelaunayEdge, 2]
                p0 = cell[edge2cell[isNonDelaunayEdge, 0], idx]
                p1 = cell[edge2cell[isNonDelaunayEdge, 0], pnext[idx]] 
                idx = edge2cell[isNonDelaunayEdge, 3]
                p2 = cell[edge2cell[isNonDelaunayEdge, 1], idx]
                p3 = cell[edge2cell[isNonDelaunayEdge, 1], pnext[idx]]
                cell[edge2cell[isNonDelaunayEdge, 0], 0] = p1
                cell[edge2cell[isNonDelaunayEdge, 0], 1] = p2
                cell[edge2cell[isNonDelaunayEdge, 0], 2] = p0

                cell[edge2cell[isNonDelaunayEdge, 1], 0] = p3
                cell[edge2cell[isNonDelaunayEdge, 1], 1] = p0
                cell[edge2cell[isNonDelaunayEdge, 1], 2] = p2

                NN = self.number_of_nodes()
                self.ds.reinit(NN, cell)

    def uniform_refine(self, n=1, surface=None, returnim=False):
        if returnim:
            nodeIMatrix = []
            cellIMatrix = []
        for i in range(n):
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            NE = self.number_of_edges()
            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')
            cell2edge = self.ds.cell_to_edge()
            edge2newNode = np.arange(NN, NN+NE)
            newNode = (node[edge[:,0],:] + node[edge[:,1],:])/2.0

            if returnim:
                A = coo_matrix((np.ones(NN), (range(NN), range(NN))), shape=(NN+NE, NN), dtype=self.ftype)
                A += coo_matrix((0.5*np.ones(NE), (range(NN, NN+NE), edge[:, 0])), shape=(NN+NE, NN), dtype=self.ftype)
                A += coo_matrix((0.5*np.ones(NE), (range(NN, NN+NE), edge[:, 1])), shape=(NN+NE, NN), dtype=self.ftype)
                nodeIMatrix.append(A.tocsr())
                B = eye(NC, dtype=self.ftype)
                B = bmat([[B], [B], [B], [B]])
                cellIMatrix.append(B.tocsr())

            if surface is not None:
                newNode, _ = surface.project(newNode)
            self.node = np.concatenate((node, newNode), axis=0)
            p = np.r_['-1', cell, edge2newNode[cell2edge]] 
            cell = np.r_['0', p[:, [0, 5, 4]], p[:, [5, 1, 3]], p[:, [4, 3, 2]], p[:, [3, 4, 5]]]
            NN = self.node.shape[0]
            self.ds.reinit(NN, cell)
        if returnim:
            return nodeIMatrix, cellIMatrix

    def uniform_bisect(self, n=1):
        for i in range(n):
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            NE = self.number_of_edges()
            node = self.node
            edge = self.ds.edge
            cell = self.ds.cell
            cell2edge = self.ds.cell_to_edge()

            cell2edge0 = np.zeros((2*NC,), dtype=self.itype)
            cell2edge0[0:NC] = cell2edge[:,0]

            edge2newNode = np.arange(NN, NN+NE)
            newNode = (node[edge[:,0],:]+node[edge[:,1],:])/2.0
            self.node = np.concatenate((node, newNode), axis=0)
            for k in range(2):
                p0 = cell[0:NC,0]
                p1 = cell[0:NC,1]
                p2 = cell[0:NC,2]
                p3 = edge2newNode[cell2edge0[0:NC]]
                cell = np.zeros((2*NC,3), dtype=self.itype)
                cell[0:NC,0] = p3 
                cell[0:NC,1] = p0 
                cell[0:NC,2] = p1 
                cell[NC:, 0] = p3 
                cell[NC:, 1] = p2 
                cell[NC:, 2] = p0 
                if k == 0:
                    cell2edge0[0:NC] = cell2edge[:,2]
                    cell2edge0[NC:] = cell2edge[:,1]
                NC = 2*NC

            NN = self.node.shape[0] 
            self.ds.reinit(NN, cell)

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NE = self.number_of_edges()

        cell = self.ds.cell
        edge = self.ds.edge
        cell2edge = self.ds.cell_to_edge()
        cell2cell = self.ds.cell_to_cell()

        isCutEdge = np.zeros((NE,), dtype=np.bool)
        while len(markedCell)>0:
            isCutEdge[cell2edge[markedCell, 0]]=True
            refineNeighbor = cell2cell[markedCell, 0]
            markedCell = refineNeighbor[~isCutEdge[cell2edge[refineNeighbor,0]]]

        edge2newNode = np.zeros((NE,), dtype=self.itype)
        edge2newNode[isCutEdge] = np.arange(NN, NN+isCutEdge.sum())

        node = self.node
        newNode =0.5*(node[edge[isCutEdge,0],:] + node[edge[isCutEdge,1],:]) 
        self.node = np.concatenate((node, newNode), axis=0)
        cell2edge0 = cell2edge[:, 0]

        for k in range(2):
            idx, = np.nonzero(edge2newNode[cell2edge0]>0)
            nc = len(idx)
            if nc == 0:
                break
            L = idx
            R = np.arange(NC, NC+nc)
            p0 = cell[idx,0]
            p1 = cell[idx,1]
            p2 = cell[idx,2]
            p3 = edge2newNode[cell2edge0[idx]]
            cell = np.concatenate((cell, np.zeros((nc,3), dtype=self.itype)), axis=0)
            cell[L,0] = p3 
            cell[L,1] = p0 
            cell[L,2] = p1 
            cell[R,0] = p3 
            cell[R,1] = p2 
            cell[R,2] = p0 
            if k == 0:
                cell2edge0 = np.zeros((NC+nc,), dtype=self.itype)
                cell2edge0[0:NC] = cell2edge[:,0]
                cell2edge0[L] = cell2edge[idx,2]
                cell2edge0[R] = cell2edge[idx,1]
            NC = NC+nc

        NN = self.node.shape[0]
        self.ds.reinit(NN, cell)

        # reconstruct the  data structure
        if u is not None:                                                       
            eu = 0.5*np.sum(u[edge[isCutEdge]], axis=1)                         
            Iu = np.concatenate((u, eu), axis=0)                                
        if u is None:                                                           
            return True                                                         
        else:                                                                   
            return(Iu, True) 


    def grad_lambda(self):
        node = self.node
        cell = self.ds.cell
        NC = self.number_of_cells()
        v0 = node[cell[:, 2], :] - node[cell[:, 1], :]
        v1 = node[cell[:, 0], :] - node[cell[:, 2], :]
        v2 = node[cell[:, 1], :] - node[cell[:, 0], :]
        dim = self.geo_dimension()
        nv = np.cross(v2, -v1)
        Dlambda = np.zeros((NC, 3, dim), dtype=self.ftype)
        if dim == 2:
            length = nv
            W = np.array([[0, 1], [-1, 0]])
            Dlambda[:,0,:] = v0@W/length.reshape((-1, 1))
            Dlambda[:,1,:] = v1@W/length.reshape((-1, 1))
            Dlambda[:,2,:] = v2@W/length.reshape((-1, 1))
        elif dim == 3:
            length = np.sqrt(np.square(nv).sum(axis=1))
            n = nv/length.reshape((-1, 1))
            Dlambda[:,0,:] = np.cross(n, v0)/length.reshape((-1,1))
            Dlambda[:,1,:] = np.cross(n, v1)/length.reshape((-1,1))
            Dlambda[:,2,:] = np.cross(n, v2)/length.reshape((-1,1))
        return Dlambda

    def jacobi_matrix(self, cellidx=None):
        """
        Return
        ------
        J : numpy.array
            `J` is the transpose o  jacobi matrix of each cell. 
            The shape of `J` is  `(NC, 2, 2)` or `(NC, 2, 3)`
        """
        node = self.node
        cell = self.ds.cell
        if cellidx is None:
            J = node[cell[:, [1, 2]]] - node[cell[:, [0]]]
        else:
            J = node[cell[cellidx, [1, 2]]] - node[cell[cellidx, [0]]]
        return J

    def rot_lambda(self):
        node = self.node
        cell = self.ds.cell
        NC = self.number_of_cells()
        v0 = node[cell[:, 2], :] - node[cell[:, 1], :]
        v1 = node[cell[:, 0], :] - node[cell[:, 2], :]
        v2 = node[cell[:, 1], :] - node[cell[:, 0], :]
        dim = self.geo_dimension()
        nv = np.cross(v2, -v1)
        Rlambda = np.zeros((NC, 3, dim), dtype=self.ftype)
        if dim == 2:
            length = nv
            Rlambda[:,0,:] = v0/length.reshape((-1, 1))
            Rlambda[:,1,:] = v1/length.reshape((-1, 1))
            Rlambda[:,2,:] = v2/length.reshape((-1, 1))
        elif dim == 3:
            length = np.sqrt(np.square(nv).sum(axis=1))
            Rlambda[:,0,:] = v0/length.reshape((-1, 1))
            Rlambda[:,1,:] = v1/length.reshape((-1, 1))
            Rlambda[:,2,:] = v2/length.reshape((-1, 1))
        return Rlambda

    def area(self, index=None):
        node = self.node
        cell = self.ds.cell
        dim = self.node.shape[1] 
        if index is None:
            v1 = node[cell[:, 1], :] - node[cell[:, 0], :]
            v2 = node[cell[:, 2], :] - node[cell[:, 0], :]
        else:
            v1 = node[cell[index, 1], :] - node[cell[index, 0], :]
            v2 = node[cell[index, 2], :] - node[cell[index, 0], :]
        nv = np.cross(v2, -v1)
        if dim == 2:
            a = nv/2.0
        elif dim == 3:
            a = np.sqrt(np.square(nv).sum(axis=1))/2.0
        return a

    def cell_area(self, index=None):
        node = self.node
        cell = self.ds.cell
        dim = self.node.shape[1] 
        if index is None:
            v1 = node[cell[:, 1], :] - node[cell[:, 0], :]
            v2 = node[cell[:, 2], :] - node[cell[:, 0], :]
        else:
            v1 = node[cell[index, 1], :] - node[cell[index, 0], :]
            v2 = node[cell[index, 2], :] - node[cell[index, 0], :]
        nv = np.cross(v2, -v1)
        if dim == 2:
            a = nv/2.0
        elif dim == 3:
            a = np.sqrt(np.square(nv).sum(axis=1))/2.0
        return a

    def bc_to_point(self, bc):
        node = self.node
        cell = self.ds.cell
        p = np.einsum('...j, ijk->...ik', bc, node[cell])
        return p 



class TriangleMeshWithInfinityNode:
    def __init__(self, mesh):
        edge = mesh.ds.edge
        bdEdgeIdx = mesh.ds.boundary_edge_index()
        NBE = len(bdEdgeIdx)
        NC = mesh.number_of_cells()
        N = mesh.number_of_nodes()

        self.itype = mesh.itype
        self.ftype = mesh.ftype

        newCell = np.zeros((NC + NBE, 3), dtype=self.itype)
        newCell[:NC, :] = mesh.ds.cell
        newCell[NC:, 0] = N 
        newCell[NC:, 1:3] = edge[bdEdgeIdx, 1::-1]

        node = mesh.node
        self.node = np.append(node, [[np.nan, np.nan]], axis=0)
        self.ds = TriangleMeshDataStructure(N+1, newCell)
        self.center = np.append(mesh.barycenter(),
                0.5*(node[edge[bdEdgeIdx, 0], :] + node[edge[bdEdgeIdx, 1], :]), axis=0)
        self.meshtype = 'tri'

    def number_of_nodes(self):
        return self.node.shape[0] 

    def number_of_edges(self):
        return self.ds.NE

    def number_of_faces(self):
        return self.ds.NC

    def number_of_cells(self):
        return self.ds.NC

    def is_infinity_cell(self):
        N = self.number_of_nodes()
        cell = self.ds.cell
        return cell[:, 0] == N-1

    def is_boundary_edge(self):
        NE = self.number_of_edges()
        cell2edge = self.ds.cell_to_edge()
        isInfCell = self.is_infinity_cell()
        isBdEdge = np.zeros(NE, dtype=np.bool)
        isBdEdge[cell2edge[isInfCell, 0]] = True
        return isBdEdge

    def is_boundary_node(self):
        N = self.number_of_nodes()
        edge = self.ds.edge
        isBdEdge = self.is_boundary_edge()
        isBdNode = np.zeros(N, dtype=np.bool)
        isBdNode[edge[isBdEdge, :]] = True
        return isBdNode

    def to_polygonmesh(self):
        isBdNode = self.is_boundary_node()
        NB = isBdNode.sum()

        nodeIdxMap = np.zeros(isBdNode.shape, dtype=self.itype)
        nodeIdxMap[isBdNode] = self.center.shape[0] + np.arange(NB)

        pnode = np.concatenate((self.center, self.node[isBdNode]), axis=0)
        PN = pnode.shape[0]

        node2cell = self.ds.node_to_cell(localidx=True)
        NV = np.asarray((node2cell > 0).sum(axis=1)).reshape(-1)
        NV[isBdNode] += 1
        NV = NV[:-1]
        
        PNC = len(NV)
        pcell = np.zeros(NV.sum(), dtype=self.itype)
        pcellLocation = np.zeros(PNC+1, dtype=self.itype)
        pcellLocation[1:] = np.cumsum(NV)


        isBdEdge = self.is_boundary_edge()
        NC = self.number_of_cells() - isBdEdge.sum()
        cell = self.ds.cell
        currentCellIdx = np.zeros(PNC, dtype=self.itype)
        currentCellIdx[cell[:NC, 0]] = range(NC)
        currentCellIdx[cell[:NC, 1]] = range(NC)
        currentCellIdx[cell[:NC, 2]] = range(NC)
        pcell[pcellLocation[:-1]] = currentCellIdx 

        currentIdx = pcellLocation[:-1]
        N = self.number_of_nodes() - 1
        currentNodeIdx = np.arange(N, dtype=self.itype)
        endIdx = pcellLocation[1:]
        cell2cell = self.ds.cell_to_cell()
        isInfCell = self.is_infinity_cell()
        pnext = np.array([1, 2, 0], dtype=self.itype)
        while True:
            isNotOK = (currentIdx + 1) < endIdx
            currentIdx = currentIdx[isNotOK]
            currentNodeIdx = currentNodeIdx[isNotOK]
            currentCellIdx = pcell[currentIdx]
            endIdx = endIdx[isNotOK]
            if len(currentIdx) == 0:
                break
            localIdx = np.asarray(node2cell[currentNodeIdx, currentCellIdx]) - 1
            cellIdx = np.asarray(cell2cell[currentCellIdx, pnext[localIdx]]).reshape(-1)
            isBdCase = isInfCell[currentCellIdx] & isInfCell[cellIdx]
            if np.any(isBdCase):
                pcell[currentIdx[isBdCase] + 1] = nodeIdxMap[currentNodeIdx[isBdCase]]
                currentIdx[isBdCase] += 1
            pcell[currentIdx + 1] = cellIdx
            currentIdx += 1

        return pnode, pcell, pcellLocation

    

