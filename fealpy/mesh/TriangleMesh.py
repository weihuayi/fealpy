import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye
from .Mesh2d import Mesh2d, Mesh2dDataStructure


class TriangleMeshDataStructure(Mesh2dDataStructure):
    localEdge = np.array([(1, 2), (2, 0), (0, 1)])
    V = 3
    E = 3
    F = 1
    def __init__(self, N, cell):
        super(TriangleMeshDataStructure, self).__init__(N, cell) 

class TriangleMesh(Mesh2d):
    def __init__(self, point, cell, dtype=np.float):

        self.point = point
        N = point.shape[0]
        self.ds = TriangleMeshDataStructure(N, cell)
        self.meshType = 'tri'
        self.dtype = dtype
        self.cellData = {}
        self.pointData = {}

    def circumcenter(self):
        point = self.point
        cell = self.ds.cell
        dim = self.geom_dimension()

        v0 = point[cell[:,2],:] - point[cell[:,1],:]
        v1 = point[cell[:,0],:] - point[cell[:,2],:]
        v2 = point[cell[:,1],:] - point[cell[:,0],:]
        nv = np.cross(v2, -v1)
        length = np.sqrt(np.square(nv).sum(axis=1))
        area = length/2.0 
        if dim == 2:
            x2 = np.sum(point**2, axis=1, keepdims=True)
            w0 = x2[cell[:,2]] + x2[cell[:,1]]
            w1 = x2[cell[:,0]] + x2[cell[:,2]]
            w2 = x2[cell[:,1]] + x2[cell[:,0]]
            W = np.array([[0, -1],[1, 0]])
            fe0 = w0*v0@W 
            fe1 = w1*v1@w 
            fe1 = w2*v2@w 
            c = 0.25*(fe0 + fe1 + fe2)/area.reshape(-1,1)
            R = np.sqrt(np.sum((c-point[cell[:,0], :])**2,axis=1))
        elif dim == 3:
            n = nv/length.reshape((-1, 1))
            l02 = np.sum(v1**2, axis=1, keepdims=True)
            l01 = np.sum(v2**2, axis=1, keepdims=True)
            d = 0.5*(l02*np.cross(n, v2) + l01*np.cross(-v1, n))/length.reshape(-1, 1)
            c = point[cell[:, 0]] + d
            R = np.sqrt(np.sum(d**2, axis=1))
        return c, R

    def angle(self):
        NC = self.number_of_cells()
        cell = self.ds.cell
        point = self.point
        localEdge = self.ds.local_edge()
        angle = np.zeros((NC, 3), dtype=np.float)
        for i,(j,k) in zip(range(3),localEdge):
            v0 = point[cell[:,j],:] - point[cell[:,i],:]
            v1 = point[cell[:,k],:] - point[cell[:,i],:]
            angle[:,i] = np.arccos(np.sum(v0*v1,axis=1)
                    /np.sqrt(np.sum(v0**2,axis=1) * np.sum(v1**2,axis=1)))
        return angle

    def edge_swap(self):
        while True:
            # Construct necessary data structure
            edge2cell = self.ds.edge_to_cell()
            cell2edge = self.ds.cell_to_edge()

            # Find non-Delaunay edges
            angle = self.angle()
            asum = np.sum(angle[edge2cell[:, 0:2], edge2cell[:, 2:4]], axis=1)
            isNonDelaunayEdge = (asum > np.pi) \
                    & (edge2cell[:,0] != edge2cell[:,1])
            if np.any(isNonDelaunayEdge) is not True:
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

                N = self.number_of_points()
                self.ds.reinit(N, cell)

    def uniform_refine(self, n=1, surface=None):
        for i in range(n):
            N = self.number_of_points()
            NC = self.number_of_cells()
            NE = self.number_of_edges()
            point = self.point
            edge = self.ds.edge
            cell = self.ds.cell
            cell2edge = self.ds.cell_to_edge()
            edge2newPoint = np.arange(N, N+NE)
            newPoint = (point[edge[:,0],:]+point[edge[:,1],:])/2.0
            if surface is not None:
                #TODO: just project the boundary point
                newPoint, _ = surface.project(newPoint)
            self.point = np.concatenate((point, newPoint), axis=0)
            p = np.concatenate((cell, edge2newPoint[cell2edge]), axis=1)
            cell = np.concatenate((
                p[:, [0, 5, 4]], 
                p[:, [5, 1, 3]],
                p[:, [4, 3, 2]],
                p[:, [3, 4, 5]]))

            N = self.number_of_points()
            self.ds.reinit(N, cell)

    def uniform_bisect(self, n=1):
        for i in range(n):
            N = self.number_of_points()
            NC = self.number_of_cells()
            NE = self.number_of_edges()
            point = self.point
            edge = self.ds.edge
            cell2edge = self.ds.cell_to_edge()

            cell2edge0 = np.zeros((2*NC,),dtype=np.int)
            cell2edge0[0:NC] = cell2edge[:,0]

            edge2newPoint = np.arange(N, N+NE)
            newPoint = (point[edge[:,0],:]+point[edge[:,1],:])/2.0
            self.point = np.concatenate((point, newPoint), axis=0)
            for k in range(2):
                p0 = cell[0:NC,0]
                p1 = cell[0:NC,1]
                p2 = cell[0:NC,2]
                p3 = edge2newPoint[cell2edge0[0:NC]]
                cell = np.zeros((2*NC,3),dtype=np.int)
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

            N = self.number_of_points()
            self.ds.reinit(N, cell)

    def bisect(self, markedCell):

        N = self.number_of_points()
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

        edge2newPoint = np.zeros((NE,),dtype=np.int)
        edge2newPoint[isCutEdge] = np.arange(N, N+isCutEdge.sum())

        point = self.point
        newPoint =0.5*(point[edge[isCutEdge,0],:] + point[edge[isCutEdge,1],:]) 
        self.point = np.concatenate((point, newPoint), axis=0)
        cell2edge0 = cell2edge[:, 0]

        for k in range(2):
            idx, = np.nonzero(edge2newPoint[cell2edge0]>0)
            nc = len(idx)
            if nc == 0:
                break
            L = idx
            R = np.arange(NC, NC+nc)
            p0 = cell[idx,0]
            p1 = cell[idx,1]
            p2 = cell[idx,2]
            p3 = edge2newPoint[cell2edge0[idx]]
            cell = np.concatenate((cell, np.zeros((nc,3),dtype=np.int)), axis=0)
            cell[L,0] = p3 
            cell[L,1] = p0 
            cell[L,2] = p1 
            cell[R,0] = p3 
            cell[R,1] = p2 
            cell[R,2] = p0 
            if k == 0:
                cell2edge0 = np.zeros((NC+nc,), dtype=np.int)
                cell2edge0[0:NC] = cell2edge[:,0]
                cell2edge0[L] = cell2edge[idx,2]
                cell2edge0[R] = cell2edge[idx,1]
            NC = NC+nc

        # reconstruct the  data structure
        N = self.number_of_points()
        self.ds.reinit(N, cell)

    def grad_lambda(self):
        point = self.point
        cell = self.ds.cell
        NC = self.number_of_cells()
        v0 = point[cell[:, 2], :] - point[cell[:, 1], :]
        v1 = point[cell[:, 0], :] - point[cell[:, 2], :]
        v2 = point[cell[:, 1], :] - point[cell[:, 0], :]
        dim = self.geom_dimension()
        nv = np.cross(v2, -v1)
        Dlambda = np.zeros((NC, 3, dim), dtype=self.dtype)
        if dim == 2:
            length = nv
            W = np.array([[0, 1], [-1, 0]], dtype=self.dtype)
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
            The shape of `J` is  `(NC, 2, 2)` or `(NC, 2, 3)`
        """
        if cellidx is None:
            J = self.point[cell[:, [1, 2]]] - mesh.point[cell[:, [0]]]
        else:
            J = self.point[cell[cellidx, [1, 2]]] - mesh.point[cell[cellidx, [0]]]
        return J

    def rot_lambda(self):
        point = self.point
        cell = self.ds.cell
        NC = self.number_of_cells()
        v0 = point[cell[:, 2], :] - point[cell[:, 1], :]
        v1 = point[cell[:, 0], :] - point[cell[:, 2], :]
        v2 = point[cell[:, 1], :] - point[cell[:, 0], :]
        dim = self.geom_dimension()
        nv = np.cross(v2, -v1)
        Rlambda = np.zeros((NC, 3, dim), dtype=self.dtype)
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

    def area(self):
        point = self.point
        cell = self.ds.cell
        v0 = point[cell[:, 2], :] - point[cell[:, 1], :]
        v1 = point[cell[:, 0], :] - point[cell[:, 2], :]
        v2 = point[cell[:, 1], :] - point[cell[:, 0], :]
        dim = self.point.shape[1] 
        nv = np.cross(v2, -v1)
        if dim == 2:
            a = nv/2.0
        elif dim == 3:
            a = np.sqrt(np.square(nv).sum(axis=1))/2.0
        return a

    def bc_to_point(self, bc):
        point = self.point
        cell = self.ds.cell
        p = np.einsum('...j, ijk->...ik', bc, point[cell])
        return p 



class TriangleMeshWithInfinityPoint:
    def __init__(self, mesh):
        edge = mesh.ds.edge
        bdEdgeIdx = mesh.ds.boundary_edge_index()
        NBE = len(bdEdgeIdx)
        NC = mesh.number_of_cells()
        N = mesh.number_of_points()

        newCell = np.zeros((NC + NBE, 3), dtype=np.int)
        newCell[:NC, :] = mesh.ds.cell
        newCell[NC:, 0] = N 
        newCell[NC:, 1:3] = edge[bdEdgeIdx, 1::-1]

        point = mesh.point
        self.point = np.append(point, [[np.nan, np.nan]], axis=0)
        self.ds = TriangleMeshDataStructure(N+1, newCell)
        self.center = np.append(mesh.barycenter(),
                0.5*(point[edge[bdEdgeIdx, 0], :] + point[edge[bdEdgeIdx, 1], :]), axis=0)
        self.meshtype = 'tri'
        self.dtype = mesh.dtype

    def number_of_points(self):
        return self.point.shape[0] 

    def number_of_nodes(self):
        return self.point.shape[0]

    def number_of_edges(self):
        return self.ds.NE

    def number_of_faces(self):
        return self.ds.NC

    def number_of_cells(self):
        return self.ds.NC

    def geom_dimension(self):
        return self.point.shape[1]

    def is_infinity_cell(self):
        N = self.number_of_points()
        cell = self.ds.cell
        return cell[:, 0] == N-1

    def is_boundary_edge(self):
        NE = self.number_of_edges()
        cell2edge = self.ds.cell_to_edge()
        isInfCell = self.is_infinity_cell()
        isBdEdge = np.zeros(NE, dtype=np.bool)
        isBdEdge[cell2edge[isInfCell, 0]] = True
        return isBdEdge

    def is_boundary_point(self):
        N = self.number_of_points()
        edge = self.ds.edge
        isBdEdge = self.is_boundary_edge()
        isBdPoint = np.zeros(N, dtype=np.bool)
        isBdPoint[edge[isBdEdge, :]] = True
        return isBdPoint

    def to_polygonmesh(self):
        isBdPoint = self.is_boundary_point()
        NBP = isBdPoint.sum()

        pointIdxMap = np.zeros(isBdPoint.shape, dtype=np.int)
        pointIdxMap[isBdPoint] = self.center.shape[0] + np.arange(NBP)

        ppoint = np.concatenate((self.center, self.point[isBdPoint]), axis=0)
        PN = ppoint.shape[0]

        point2cell = self.ds.point_to_cell(localidx=True)
        NV = np.asarray((point2cell > 0).sum(axis=1)).reshape(-1)
        NV[isBdPoint] += 1
        NV = NV[:-1]
        
        PNC = len(NV)
        pcell = np.zeros(NV.sum(), dtype=np.int)
        pcellLocation = np.zeros(PNC+1, dtype=np.int)
        pcellLocation[1:] = np.cumsum(NV)


        isBdEdge = self.is_boundary_edge()
        NC = self.number_of_cells() - isBdEdge.sum()
        cell = self.ds.cell
        currentCellIdx = np.zeros(PNC, dtype=np.int)
        currentCellIdx[cell[:NC, 0]] = range(NC)
        currentCellIdx[cell[:NC, 1]] = range(NC)
        currentCellIdx[cell[:NC, 2]] = range(NC)
        pcell[pcellLocation[:-1]] = currentCellIdx 

        currentIdx = pcellLocation[:-1]
        N = self.number_of_points() - 1
        currentPointIdx = np.arange(N)
        endIdx = pcellLocation[1:]
        cell2cell = self.ds.cell_to_cell()
        isInfCell = self.is_infinity_cell()
        pnext = np.array([1, 2, 0], dtype=np.int)
        while True:
            isNotOK = (currentIdx + 1) < endIdx
            currentIdx = currentIdx[isNotOK]
            currentPointIdx = currentPointIdx[isNotOK]
            currentCellIdx = pcell[currentIdx]
            endIdx = endIdx[isNotOK]
            if len(currentIdx) == 0:
                break
            localIdx = np.asarray(point2cell[currentPointIdx, currentCellIdx]) - 1
            cellIdx = np.asarray(cell2cell[currentCellIdx, pnext[localIdx]]).reshape(-1)
            isBdCase = isInfCell[currentCellIdx] & isInfCell[cellIdx]
            if np.any(isBdCase):
                pcell[currentIdx[isBdCase] + 1] = pointIdxMap[currentPointIdx[isBdCase]]
                currentIdx[isBdCase] += 1
            pcell[currentIdx + 1] = cellIdx
            currentIdx += 1

        return ppoint, pcell, pcellLocation

    

