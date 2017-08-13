import numpy as np


class UnitIntervalSpace():

    def __init__(self, p=1, dtype=np.float):
        self.p = p
        self.interval_idx_matrix()

    def interval_idx_matrix(self):
        p = self.p
        intervalIdx = np.zeros((p+1, 2), dtype=np.int)
        intervalIdx[:, 0] = np.arange(p, -1, -1)
        intervalIdx[:, 1] = p - intervalIdx[:, 0]
        self.intervalIdx =  intervalIdx

    def basis(self, x):

        bc = np.array([1-x, x], dtype=np.float)

        p = self.p
        pp = p**p

        ldof = self.number_of_local_dofs() 
        phi = np.zeros((ldof, ), dtype=self.dtype)

        P = np.ones((p+1,), dtype=np.float)
        A = np.ones((p+1, 2), dtype=np.float)

        c = np.arange(1, p+1, dtype=np.int)
        t = np.linspace(0, 1, p, endpoint=False).reshape((-1,1))

        P[1:] = 1.0/np.multiply.accumulate(c)
        A[1:,:] = bc - t
        B = P.reshape(-1, 1)*np.multiply.accumulate(A, axis=0)
        phi = pp*np.prod(B[self.intervalIdx, [0, 1]], axis=1)

        return phi

    def grad_basis(self, x):

        bc = np.array([1-x, x], dtype=np.float)

        p = self.p
        pp = p**p

        P = np.ones((p+1,), dtype=np.float)
        A = np.ones((p+1, 2), dtype=np.float)

        c = np.arange(1, p+1, dtype=np.int)
        t = np.linspace(0, 1, p, endpoint=False).reshape((-1,1))

        P[1:] = 1.0/np.multiply.accumulate(c)
        A[1:,:] = bc - t

        B = P.reshape(-1,1)*np.multiply.accumulate(A, axis=0)

        F = np.zeros((p+1, 2), dtype=self.dtype)
        for i in range(2):
            Fi = A[1:, [i]] - np.diag(A[1:,i]) + np.eye(p)
            F[1:, i] = np.tril(np.multiply.accumulate(Fi, axis=0)).sum(axis=1)
        F *= P.reshape(-1,1)

        Q = B[self.intervalIdx, [0, 1]]
        M = F[self.intervalIdx, [0, 1]]
        gradphi = -M[:, 0]*Q[:, 1] + Q[:, 0]*M[:, 1]
        return gradphi


class QuadrangleFiniteElementSpace():

    def __init__(self, mesh, p=1, dtype=np.float):

        self.mesh = mesh
        self.p = p
        self.dtype= dtype
        self.uispace = UnitIntervalSpace(p=p, dtype=np.float)
        J, I = np.meshgrid(range(p+1), range(p+1))
        self.I = I
        self.J = J


    def __str__(self):
        return "Lagrange finite element space on Quadrangle mesh!"

    def linear_basis(self, bc):
        J, I= np.meshgrid([1-bc[1], bc[1]], [1-bc[0], bc[0]])  
        phi = (I*J).reshape(-1)
        return phi

    def basis(self, bc):
        # bc=[\xi, \eta] is in reference elment 

        phi0 = self.uispace.basis(bc[0])
        phi1 = self.uispace.basis(bc[1])

        I, J = self.I, self.J
        phi = (phi0[I]*phi1[J]).reshape(-1)
        return phi

    def grad_basis(self, bc):
        phi0 = self.uispace.basis(bc[0])
        phi1 = self.uispace.basis(bc[1])

        gradphi0 = self.uispace.grad_basis(bc[0])
        gradphi1 = self.uispace.grad_basis(bc[1])

        ldof = self.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        
        I, J = self.I, self.J
        gradphi = np.zeros((ldof, 2), dtype=np.float)
        gradphi[:, 0] = (gradphi0[I]*phi1[J]).reshape(-1)
        gradphi[:, 1] = (phi0[I]*gradphi1[J]).reshape(-1)
        invJ, detJ = self.inv_jacobi_matrix(bc)

        # Here the graddients of all basis functions are the real graphi multiplied with the det of Jacobi
        # matrix
        gradphi = np.einsum('ij, kj...->ki...', gradphi, invJ)

        return gradphi, detJ


    def inv_jacobi_matrix(self, bc):
        mesh = self.mesh
        
        point = mesh.point
        cell = mesh.ds.cell
        NC = mesh.number_of_cells()

        # The local idx for the basises is different with the local idx in mesh
        # x1---------x3
        # |          |
        # |          |
        # x0---------x2

        x0 = point[cell[:, 0]]
        x1 = point[cell[:, 3]]
        x2 = point[cell[:, 1]]
        x3 = point[cell[:, 2]]

        J = np.zeros((NC, 2, 2), dtype=np.float)
        t = x0 - x1 - x2 + x3
        J[:, :, 0] = x2 - x0 + bc[1]*t
        J[:, :, 1] = x1 - x0 + bc[0]*t

        # Here we return the inv of Jacobi matrix multiplied with its det 
        return np.linalg.inv(J), np.linalg.det(J)

    def edge_to_dof(self):
        p = self.p
        
        mesh = self.mesh
        N = mesh.number_of_points()
        NE = mesh.number_of_edges()

        edge = mesh.ds.edge
        edge2dof = np.zeros((NE, p+1), dtype=np.int) 
        edge2dof[:, [0, -1]] = edge 
        if p > 1:
            edge2dof[:, 1:-1] = N + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def edge_dof(self):
        p = self.p
        fdof = (p+1)*(p+1) 
        edgeDof = np.zeros((4, p+1), dtype=np.int)
        edgeDof[0, :] = range(0, p*(p+1)+1, p+1)
        edgeDof[1, :] = range(p*(p+1), fdof)
        edgeDof[2, :] = range(p, fdof, p+1) 
        edgeDof[3, :] = range(p+1)
        return edgeDof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        edge = mesh.ds.edge

        N = mesh.number_of_points()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()

        cell2dof = np.zeros((NC, ldof), dtype=np.int)

        edge2dof = self.edge_to_dof()
        cell2edge = mesh.ds.cell_to_edge()

        edgeIdx = np.zeros((2, p+1), dtype=np.int)
        edgeIdx[0, :] = range(p+1)
        edgeIdx[1, :] = edgeIdx[0, -1::-1]

        fe = np.array([0, 1, 3, 0])
        edgeDof = self.edge_dof()
        for i in range(4):
            I = np.ones(NC, dtype=np.int)
            sign = (cell[:, fe[i]] == edge[cell2edge[:, i], 0])
            I[sign] = 0
            cell2dof[:, edgeDof[i, :]] = edge2dof[cell2edge[:, [i]], edgeIdx[I]]

        if p > 1:
            base = N + (p-1)*NE
            isInCellDof = np.ones(ldof, dtype=np.bool)
            isInCellDof[edgeDof] = False
            idof = ldof - 4*p
            cell2dof[:, isInCellDof] = base + np.arange(NC*idof).reshape(NC, idof)

        return cell2dof

    def number_of_local_dofs(self):
        p = self.p
        return (p+1)*(p+1)

    def number_of_global_dofs(self):
        p = self.p

        mesh = self.mesh
        N = mesh.number_of_points()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        gdof = N

        if p > 1:
            gdof += NE*(p-1)
            gdof += (ldof - 4 - 4*(p-1))*NC

        return gdof


    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        point = mesh.point

        N = point.shape[0]
        NC = mesh.number_of_cells() 

        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        ipoints = np.zeros((gdof, 2), dtype=np.float)

        xi = np.linspace(0, 1, p+1)
        bc = np.zeros((ldof, 2), dtype=np.float)
        bc[:, 0] = xi[self.I].reshape(-1)
        bc[:, 1] = xi[self.J].reshape(-1)

        idx = np.array([0, 3, 1, 2])

        cell2dof = self.cell_to_dof()

        for i in range(ldof):
            phi = self.linear_basis(bc[i, :])
            ipoints[cell2dof[:, i]] = np.einsum('j, kj...->k...', phi, point[cell[:, idx]])

        return ipoints

class HexahedronFiniteElementSpace():

    def __init__(self, mesh, p=1, dtype=np.float):
        self.mesh = mesh
        self.p = p
        self.dtype = dtype
        self.uispace = UnitIntervalSpace(p=p, dtype=np.float)
        J, I, K = np.meshgrid(range(p+1), range(p+1), range(p+1))
        self.I = I
        self.J = J
        self.K = K

    def __str__(self):
        return "Lagrange finite element space on Hex mesh!"

    def linear_basis(self, bc):
        J, I, K = np.meshgrid([1-bc[1], bc[1]], [1-bc[0], bc[0]], [1-bc[2], bc[2]])  
        phi = (I*J*K).reshape(-1)
        return phi


    def basis(self, bc):
        # bc=[\xi, \eta, \beta] is in reference elment 

        phi0 = self.uispace.basis(bc[0])
        phi1 = self.uispace.basis(bc[1])
        phi2 = self.uispace.basis(bc[2])

        I, J, K = self.I, self.J, self.K
        phi = (phi0[I]*phi1[J]*phi2[K]).reshape(-1) 

        return phi

    def grad_basis(self, bc):
        phi0 = self.uispace.basis(bc[0])
        phi1 = self.uispace.basis(bc[1])
        phi2 = self.uispace.basis(bc[2])

        gradphi0 = self.uispace.grad_basis(bc[0])
        gradphi1 = self.uispace.grad_basis(bc[1])
        gradphi2 = self.uispace.grad_basis(bc[2])

        ldof = self.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        
        I, J, K = self.I, self.J, self.K
        gradphi = np.zeros((ldof, 3), dtype=np.float)
        gradphi[:, 0] = (gradphi0[I]*phi1[J]*phi2[K]).reshape(-1)
        gradphi[:, 1] = (phi0[I]*gradphi1[J]*phi2[K]).reshape(-1)
        gradphi[:, 2] = (phi0[I]*phi1[J]*gradphi2[K]).reshape(-1)
        invJ = self.inv_jacobi_matrix(bc)

        # Here the graddients of all basis functions are the real graphi multiplied with the det of Jacobi
        # matrix
        gradphi = np.einsum('ij, kj...->ki...', gradphi, invJ)

        return gradphi



    def inv_jacobi_matrix(self, bc):
        mesh = self.mesh
        
        point = mesh.point
        cell = mesh.ds.cell
        NC = mesh.number_of_cells()

        x0 = point[cell[:, 0]]
        x1 = point[cell[:, 4]]
        x2 = point[cell[:, 3]]
        x3 = point[cell[:, 7]]

        x4 = point[cell[:, 1]]
        x5 = point[cell[:, 5]]
        x6 = point[cell[:, 2]]
        x7 = point[cell[:, 6]]

        J = np.zeros((NC, 3, 3), dtype=np.float)
        t0 = x0 - x1 + x2 - x3 + x4 - x5 + x7 - x6
        t1 = x0 - x4 + x6 - x2
        t2 = x0 - x2 + x3 - x1
        t3 = x0 - x1 + x5 - x4

        J[:, :, 0] = x4 - x0 + bc[1]*bc[2]*t0 + bc[1]*t1 + bc[2]*t3 
        J[:, :, 1] = x2 - x0 + bc[2]*bc[0]*t0 + bc[2]*t2 + bc[0]*t1
        J[:, :, 2] = x1 - x0 + bc[0]*bc[1]*t0 + bc[0]*t3 + bc[1]*t2

        # Here we return the inv of Jacobi matrix multiplied with its det 
        return np.linalg.det(J).reshape(-1, 1, 1)*np.linalg.inv(J)

    def edge_to_dof(self):
        p = self.p
        
        mesh = self.mesh
        N = mesh.number_of_points()
        NE = mesh.number_of_edges()

        edge = mesh.ds.edge
        edge2dof = np.zeros((NE, p+1), dtype=np.int) 
        edge2dof[:, [0, -1]] = edge 
        if p > 1:
            edge2dof[:, 1:-1] = N + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def edge_dof(self):
        p = self.p
        fdof = (p+1)*(p+1) 
        edgeDof = np.zeros((4, p+1), dtype=np.int)
        edgeDof[0, :] = range(0, p*(p+1)+1, p+1)
        edgeDof[1, :] = range(p*(p+1), fdof)
        edgeDof[2, :] = range(p, fdof, p+1) 
        edgeDof[3, :] = range(p+1)
        return edgeDof

    def face_to_dof(self):
        p =self.p
        mesh = self.mesh

        N = mesh.number_of_points()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()

        fdof = (p+1)*(p+1)

        face = mesh.ds.face
        edge = mesh.ds.edge
        face2dof = np.zeros((NF, fdof), dtype=np.int)

        edge2dof = self.edge_to_dof()
        face2edge = mesh.ds.face_to_edge()

        edgeIdx = np.zeros((2, p+1), dtype=np.int)
        edgeIdx[0, :] = range(p+1)
        edgeIdx[1, :] = edgeIdx[0, -1::-1]

        fe = np.array([0, 1, 3, 0])
        edgeDof = self.edge_dof()
        for i in range(4):
            I = np.ones(NF, dtype=np.int)
            sign = (face[:, fe[i]] == edge[face2edge[:, i], 0])
            I[sign] = 0
            face2dof[:, edgeDof[i, :]] = edge2dof[face2edge[:, [i]], edgeIdx[I]]

        if p > 1:
            base = N + (p-1)*NE
            isInFaceDof = np.ones(fdof, dtype=np.bool)
            isInFaceDof[edgeDof] = False
            fidof = fdof - 4*p
            face2dof[:, isInFaceDof] = base + np.arange(NF*fidof).reshape(NF, fidof)

        return face2dof

    def face_dof(self):
        p = self.p
        fdof = (p+1)*(p+1) 
        faceDof = np.zeros((6, fdof), dtype=np.int)
        faceDof[0, :] = (np.arange(0, p*(p+1)+1, p+1) + np.arange(0, (p+1)*fdof, fdof).reshape(-1, 1)).reshape(-1)
        faceDof[1, :] = (np.arange(p, fdof, p+1) + np.arange(0, (p+1)*fdof, fdof).reshape(-1, 1)).reshape(-1)
        faceDof[2, :] = (np.arange(p+1) + np.arange(0, fdof, p+1).reshape(-1, 1)).reshape(-1) 
        faceDof[3, :] = faceDof[2, :] + p*fdof
        faceDof[4, :] = (np.arange(p+1) + np.arange(0, (p+1)*fdof, fdof).reshape(-1, 1)).reshape(-1)
        faceDof[5, :] = (np.arange(p*(p+1), fdof) + np.arange(0, (p+1)*fdof, fdof).reshape(-1, 1)).reshape(-1)
        return faceDof


    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        face = mesh.ds.face

        N = mesh.number_of_points()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        fdof = (p+1)*(p+1)

        localFace = np.array([
            [0, 1, 2, 3], [4, 5, 6, 7], 
            [0, 3, 7, 4], [1, 2, 6, 5], 
            [0, 1, 5, 4], [3, 2, 6, 7]], dtype=np.int) 


        cell2dof = np.zeros((NC, ldof), dtype=np.int)

        cell2face = mesh.ds.cell_to_face()
        face2dof = self.face_to_dof()
        faceDof = self.face_dof()

        dofIdx0 = np.arange(fdof).reshape(p+1, p+1)

        dofIdx = np.zeros((8, fdof), dtype=np.int)

        for i in range(4):
            dofIdx[i, :] = np.rot90(dofIdx0, k=i).reshape(-1)
            dofIdx[i+4, :] = np.rot90(dofIdx0.T, k=i).reshape(-1)

        ne = np.array([1, 2, 3, 0])
        for i in range(6):
            fi = face[cell2face[:, i]]
            fj = cell[:, localFace[i]]
            idxj = np.argsort(fj, axis=1)
            idxjr = np.argsort(idxj, axis=1)
            idxi = np.argsort(fi, axis=1)
            idx = idxi[np.arange(NC).reshape(-1, 1), idxjr] # fj = fi[:, idx]
            I = idx[:, 0]
            I[idx[:, 1] != ne[idx[:, 0]]] += 4
            cell2dof[:, faceDof[i, :]] = face2dof[cell2face[:, [i]], dofIdx[I]]

        if p > 1:
            fidof = fdof - 4*p
            base = N + (p-1)*NE + NF*fidof
            isInCellDof = np.ones(ldof, dtype=np.bool)
            isInCellDof[faceDof] = False
            cidof = ldof - 8 - 12*(p-1) - 6*fidof
            cell2dof[:, isInCellDof] = base + np.arange(NC*cidof).reshape(NC, cidof)

        return cell2dof

    def number_of_local_dofs(self):
        p = self.p
        return (p+1)*(p+1)*(p+1)

    def number_of_global_dofs(self):

        p = self.p
        mesh = self.mesh
        
        N = mesh.number_of_points()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()

        gdof = N

        if p>1:
            fidof = (p+1)*(p+1) - 4*p
            cidof = ldof - 8 - 12*(p-1) - 6*fidof
            gdof += NE*(p-1) + NF*fidof + NC*cidof

        return gdof


    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        point = mesh.point

        N = point.shape[0]
        NC = mesh.number_of_cells() 

        ldof = self.number_of_local_dofs()
        gdof = self.number_of_global_dofs()
        ipoints = np.zeros((gdof, 3), dtype=np.float)

        xi = np.linspace(0, 1, p+1)
        bc = np.zeros((ldof, 3), dtype=np.float)
        bc[:, 0] = xi[self.I].reshape(-1)
        bc[:, 1] = xi[self.J].reshape(-1)
        bc[:, 2] = xi[self.K].reshape(-1)


        idx = np.array([0, 4, 3, 7, 1, 5, 2, 6])

        cell2dof = self.cell_to_dof()

        for i in range(ldof):
            phi = self.linear_basis(bc[i, :])
            ipoints[cell2dof[:, i]] = np.einsum('j, kj...->k...', phi, point[cell[:, idx]])

        return ipoints



    def interpolation(self, u, uI):
        ipoint = self.interpolation_points()
        uI = FiniteElementFunction(self)
        uI[:] = u(ipoint)
        return uI
