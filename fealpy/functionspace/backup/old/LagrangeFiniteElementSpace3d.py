import numpy as np
from ..common import ranges

class LagrangeFiniteElementSpace3d:
    def __init__(self, mesh, p=1, dtype=np.float):
        self.mesh = mesh
        self.p = p 
        self.dtype=dtype
        self.cellIdx = self.cell_idx_matrix()
        self.faceIdx = self.face_idx_matrix()

    def __str__(self):
        return "Lagrange finite element space on tet mesh!"

    def cell_idx_matrix(self):
        p = self.p
        ldof = self.number_of_local_dofs()
        cidx = np.zeros((4, ldof), dtype=np.int)
        cidx[0,:] = np.repeat(range(p, -1, -1), np.cumsum(range(1, p+2)))
        cidx[3,:] = ranges(ranges(range(1,p+2), start=1))
        cidx[2,:] = np.repeat(ranges(range(1,p+2)), ranges(range(1,p+2), start=1)) - cidx[3,:]
        cidx[1,:] = p - cidx[0,:] - cidx[2,:] - cidx[3,:]
        return cidx

    def face_idx_matrix(self):
        p = self.p
        fdof = int((p+1)*(p+2)/2)
        fidx = np.zeros((fdof, 3), dtype=np.int)
        fidx[0,:] = np.repeat(range(p, -1, -1), range(1, p+2))
        fidx[2,:] = ranges(range(1,p+2)) 
        fidx[1,:] = p - fidx[0,:] - fidx[2,:] 
        return fidx

    def is_on_node_local_dof(self):
        p = self.p
        cellIdx = self.cellIdx
        isNodeDof = (cellIdx == p)
        return isNodeDof

    def is_on_edge_local_dof(self):
        p =self.p
        cellIdx = self.cellIdx
        ldof = self.number_of_local_dofs()
        localEdge = self.mesh.ds.localEdge
        isEdgeDof = np.zeros((ldof, 6), dtype=np.bool_)
        for i in range(6):
            isEdgeDof[i,:] = (cellIdx[localEdge[-(i+1),0],:] == 0) & (cellIdx[localEdge[-(i+1),1],:] == 0 )
        return isEdgeDof

    def is_on_face_local_dof(self):
        p = self.p
        cellIdx = self.cellIdx
        ldof = self.number_of_local_dofs()
        isFaceDof = (cellIdx == 0)
        return isFaceDof

    def basis(self, bc):
        pass

    def grad_basis(self, bc):
        pass

    def hessian_basis(self, bc):
        pass

    def dual_basis(self, u):
        pass

    def value(self, uh, bc):
        pass
    
    def grad_value(self, uh, bc):
        pass

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        pass
    
    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        NE = mesh.number_of_edges()
        N = mesh.number_of_points()

        base = N

        edge = mesh.ds.edge
        edge2dof = np.zeros((NE,p+1), dtype=np.int) 
        edge2dof[:, [0, -1]] = edge 
        if p > 1:
            edge2dof[:,1:-1] = base + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof


    def face_to_dof(self):
        p = self.p
        fdof = int((p+1)*(p+2)/2)

        mesh = self.mesh

        N = mesh.number_of_points()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        base = N + (p-1)*NE

        face = mesh.ds.face
        face2edge = mesh.ds.face_to_edge()

        edge2dof = self.edge_to_dof()

        face2dof = np.zeros((NF, fdof), dtype=np.int)
        faceIdx = self.faceIdx

        isEdgeDof = faceIdx == 0 

        face2edgeSign = mesh.ds.face_to_edge_sign()
        for i in range(3):
            face2dof[face2edgeSign[:,[i]], isEdgeDof[0,:]] = edge2dof[face2edge[face2edgeSign[:,[i]],0], :]
            face2dof[~face2edgeSign[:,[i]], isEdgeDof[0,:]]= edge2dof[face2edge[~face2edgeSign[:,[i]],0],-1::-1]

        if p > 2:
            isInFaceDof = ~(isEdgeDof[0,:] | isEdgeDof[1,:] | isEdgeDof[2,:])
            fidof = fdof - 3*p
            face2dof[:, isInFaceDof] = base + np.arange(NF*fidof).reshape(NF, fidof)
        return face2dof

    def cell_to_dof(self):

        p = self.p
        fdof = int((p+1)*(p+2)/2)
        ldof = self.number_of_local_dofs()
        mesh = self.mesh

        N = mesh.number_of_points()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        base = N + (p-1)*NE
        if p > 2:
            base += (fdof - 3*p)*NF

        localFace = np.array([(1,2,3),(0,2,3),(0,1,3),(0,1,2)], dtype=np.int)

        cell2dof = np.zeros((NC, ldof), dtype=np.int)

        face = mesh.ds.face 
        face2cell = mesh.ds.face_to_cell()

        face2dof = self.face_to_dof()
        face1 = cell[face2cell[:,[0]], localFace[face2cell[:,2]]]
        face2 = cell[face2cell[:,[1]], localFace[face2cell[:,3]]]

        isFaceDof = self.local_dof_on_face()
        cell2dof[face2cell[:,[0]], isFaceDof[face2cell[:,2],:]] = face2dof[:, ]
        cell2dof[face2cell[:,[1]], isFaceDof[face2cell[:,3],:]] = face2dof[:, ]
        return cell2dof

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        N = mesh.number_of_points()
        gdof = N

        if p > 1:
            NE = mesh.number_of_edges()
            edof = p - 1
            gdof += edof*NE

        if p > 2:
            NF = mesh.number_of_faces()
            fdof = int((p+1)*(p+2)/2) - 3*p
            gdof += fdof*NF

        if p > 3:
            NC = mesh.number_of_cells()
            ldof = self.number_of_local_dofs()
            cdof = ldof - 6*edof - 4*fdof - 4
            gdof += cdof*NC

        return gdof


    def number_of_local_dofs(self):
        p = self.p
        ldof = (p+1)*(p+2)*(p+3)/6
        return int(ldof)

    def interpolation_points(self):
        pass

    def interpolation(self, u, uI):
        pass

    def projection(self, u, up):
        pass

    def array(self):
        pass
