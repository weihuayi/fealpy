from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye
import numpy as np
from ..quadrature  import TriangleQuadrature

class LaplaceSymetricForm:
    def __init__(self, V, qfindex=1, dtype=np.float):
        self.V = V
        self.qfindex = qfindex 
        self.dtype = dtype

    def get_matrix(self):
        V = self.V

        mesh = V.mesh
        NC = mesh.number_of_cells() 
        area = mesh.area()
        
        qf = TriangleQuadrature(self.qfindex)
        nQuad = qf.get_number_of_quad_points()

        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        cell2dof = V.cell_to_dof()
        A = coo_matrix((gdof, gdof), dtype=self.dtype)
        for i in range(ldof):
            for j in range(i,ldof):
                val = np.zeros((NC,), dtype=self.dtype)
                for q in range(nQuad):
                    lambda_k, w_k = qf.get_gauss_point_and_weight(q)
                    gradphi = V.grad_basis(lambda_k)
                    val += np.sum(gradphi[:,i,:]*gradphi[:,j,:], axis=1)*w_k
                A += coo_matrix((val*area, (cell2dof[:,i], cell2dof[:,j])), shape=(gdof, gdof))
                if j != i:
                    A += coo_matrix((val*area, (cell2dof[:,j], cell2dof[:,i])), shape=(gdof, gdof))


        return A.tocsr()

class DiffusionForm:
    def __init__(self, V0, V1):
        self.V0 = V0
        self.V1 = V1

    def get_matrix(self):
        pass

class MassForm:
    def __init__(self, V, qfidx=3, dtype=np.float):
        self.V = V
        self.qfidx = qfidx 
        self.dtype=dtype

    def get_matrix(self):
        V = self.V
        mesh = V.mesh
        cell = mesh.ds.cell
        NC = mesh.number_of_cells()
        area = mesh.area()
        
        qf = TriangleQuadrature(self.qfidx)
        nQuad = qf.get_number_of_quad_points()

        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        cell2dof = V.cell_to_dof()
        M = coo_matrix((gdof, gdof), dtype=self.dtype)

        for i in range(ldof):
            for j in range(i, ldof):
                val = np.zeros(NC, dtype=self.dtype)
                for q in range(nQuad):
                    lambda_k, w_k = qf.get_gauss_point_and_weight(q)
                    phi = V.basis(lambda_k)
                    val += w_k*phi[i]*phi[j]
                val *=area
                M += coo_matrix(
                        (val, (cell2dof[:,i], cell2dof[:,j])),
                        shape=(gdof, gdof))
                if j != i:
                    M += coo_matrix(
                            (val, (cell2dof[:,j], cell2dof[:,i])), 
                            shape=(gdof, gdof))
        return M.tocsr()

class SourceForm:
    def __init__(self, V, f, qfidx=3, dtype=np.float):
        self.V = V
        self.f = f
        self.qfidx = qfidx 
        self.dtype = dtype

    def get_vector(self):
        V = self.V
        mesh = V.mesh
        f = self.f

        NC = mesh.number_of_cells()
        qf = TriangleQuadrature(self.qfidx)
        nQuad = qf.get_number_of_quad_points()

        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        cell2dof = V.cell_to_dof()
        
        bb = np.zeros((NC, ldof), dtype=self.dtype)
        area = mesh.area()
        for i in range(nQuad):
            lambda_k, w_k = qf.get_gauss_point_and_weight(i)
            p = mesh.bc_to_point(lambda_k)
            fval = f(p)
            phi = V.basis(lambda_k)
            for j in range(ldof):
                bb[:, j] += fval*phi[j]*w_k

        bb *= area.reshape(-1, 1)
        cell2dof = V.cell_to_dof()
        b = np.zeros((gdof,), dtype=self.dtype)
        np.add.at(b, cell2dof.flatten(), bb.flatten())
        #b = np.bincount(cell2dof.flatten(), weights=bb.flatten(), minlength=gdof)
        return b


class BihamonicRecoveryForm:

    def __init__(self, V, sigma=1):
        self.V = V
        self.sigma = sigma 

    def get_matrix(self):
        V = self.V

        mesh = V.mesh
        NC = mesh.number_of_cells() 
        N = mesh.number_of_points() 
        cell = mesh.ds.cell
        point = mesh.point

        edge2cell = mesh.ds.edge_to_cell()
        edge = mesh.ds.edge
        isBdEdge = (edge2cell[:,0]==edge2cell[:,1])
        bdEdge = edge[isBdEdge]
        
        W = np.array([[0, -1], [1, 0]], dtype=np.int)
        n = (point[bdEdge[:,1],] - point[bdEdge[:,0],:])@W
        h = np.sqrt(np.sum(n**2, axis=1)) 
        n /= h.reshape(-1, 1)

        A = coo_matrix((N, N), dtype=np.float)
        B = coo_matrix((N, N), dtype=np.float)
        P = coo_matrix((N, N), dtype=np.float)
        Q = coo_matrix((N, N), dtype=np.float)
        S = coo_matrix((N, N), dtype=np.float)
        T = coo_matrix((N, N), dtype=np.float)
        G = coo_matrix((N, N), dtype=np.float)
        gradphi, area = mesh.grad_lambda()
        for i in range(3):
            for j in range(3):  
                A += coo_matrix((gradphi[:,j,0], (cell[:,i], cell[:,j])), shape=(N,N))
                B += coo_matrix((gradphi[:,j,1], (cell[:,i], cell[:,j])), shape=(N,N))
                val00 = gradphi[:,i,0]*gradphi[:,j,0]*area 
                val01 = gradphi[:,i,0]*gradphi[:,j,1]*area
                val11 = gradphi[:,i,1]*gradphi[:,j,1]*area
                P += coo_matrix((val00, (cell[:,i], cell[:,j])), shape=(N, N))
                Q += coo_matrix((val01, (cell[:,i], cell[:,j])), shape=(N, N))
                S += coo_matrix((val11, (cell[:,i], cell[:,j])), shape=(N, N))

                val = np.sum(gradphi[edge2cell[isBdEdge, 0], i, :]*n, axis=1) \
                       *np.sum(gradphi[edge2cell[isBdEdge, 0], j, :]*n, axis=1)/h
                T += coo_matrix((self.sigma*val, (cell[edge2cell[isBdEdge,0], i], cell[edge2cell[isBdEdge,0],j])), shape=(N,N))


        D = spdiags(1.0/np.bincount(cell.flatten()), 0, N, N)
        A = D@A.tocsc()
        B = D@B.tocsc()
        P = P.tocsc()
        Q = Q.tocsc()
        S = S.tocsc()
        T = T.tocsc()

        M = A.transpose()@P@A + A.transpose()@Q@B + B.transpose()@Q.transpose()@A+B.transpose()@S@B + T

        bdCellIdx = edge2cell[isBdEdge, 0]
        bdCell = cell[bdCellIdx]
        NBE = len(bdEdge) 
        VM = csr_matrix((NBE, N))
        for i in range(3):
            VM += spdiags(gradphi[bdCellIdx, i, 0], 0, NBE, NBE) @ A[bdCell[:, i], :] 
            VM += spdiags(gradphi[bdCellIdx, i, 1], 0, NBE, NBE) @ B[bdCell[:, i], :]

        Y = coo_matrix((N, N), dtype=np.float)
        for i in range(3):
            VAL = spdiags(h*np.sum(gradphi[bdCellIdx, i, :]*n, axis=1), 0, NBE, NBE)@VM
            I, J = VAL.nonzero()
            val = np.asarray(VAL[I, J]).reshape(-1)
            Y += coo_matrix((val, (bdCell[I, i], J)), shape=(N, N)) 

#        M -= Y.tocsr()

#        nn = np.array([1, 2, 0])
#        idx0 = edge2cell[isBdEdge, 2]
#        idx1 = nn[idx0]
#        idx2 = nn[idx1]
#
#        gidx = (cell[bdCellIdx, idx0], cell[bdCellIdx, idx1], cell[bdCellIdx, idx2])

#        nn = np.array([1, 2, 0])
#        idx0 = edge2cell[isBdEdge, 2]
#        idx1 = nn[idx0]
#        idx2 = nn[idx1]
#
#        gidx0 = cell[bdCellIdx, idx0]
#        gidx1 = cell[bdCellIdx, idx1]
#        gidx2 = cell[bdCellIdx, idx2]
#
#        VAL0 = np.asarray(A[gidx1, gidx0]).reshape(-1) + np.asarray(A[gidx2, gidx0]).reshape(-1)
#        VAL1 = np.asarray(B[gidx1, gidx0]).reshape(-1) + np.asarray(B[gidx2, gidx0]).reshape(-1)
#        VAL = spdiags(0.5*h*(VAL0*n[:, 0] + VAL1*n[:, 1]), 0, NBE, NBE)*VM
#        I, J = VAL.nonzero()
#        val = np.asarray(VAL[I, J]).reshape(-1)
#        I = bdCell[np.arange(NBE), idx0][I]
#        Y = coo_matrix((val, (I, J)), shape=(N, N))
#
#        VAL0 = np.asarray(A[gidx1, gidx1]).reshape(-1) + np.asarray(A[gidx2, gidx1]).reshape(-1)
#        VAL1 = np.asarray(B[gidx1, gidx1]).reshape(-1) + np.asarray(B[gidx2, gidx1]).reshape(-1)
#        VAL = spdiags(0.5*h*(VAL0*n[:, 0] + VAL1*n[:, 1]), 0, NBE, NBE)*VM
#        I, J = VAL.nonzero()
#        val = np.asarray(VAL[I, J]).reshape(-1)
#        I = bdCell[np.arange(NBE), idx1][I]
#        Y += coo_matrix((val, (I, J)), shape=(N, N))
#
#        VAL0 = np.asarray(A[gidx2, gidx2]).reshape(-1) + np.asarray(A[gidx1, gidx2]).reshape(-1)
#        VAL1 = np.asarray(B[gidx2, gidx2]).reshape(-1) + np.asarray(B[gidx1, gidx2]).reshape(-1)
#        VAL = spdiags(0.5*h*(VAL0*n[:, 0] + VAL1*n[:, 1]), 0, NBE, NBE)*VM
#        I, J = VAL.nonzero()
#        val = np.asarray(VAL[I, J]).reshape(-1)
#        I = bdCell[np.arange(NBE), idx2][I]
#        Y += coo_matrix((val, (I, J)), shape=(N, N))
#        M -= Y.tocsr()

        # u_nv_n
        #AB0 = spdiags(n[:, 0], 0, NBE, NBE)


        return M
