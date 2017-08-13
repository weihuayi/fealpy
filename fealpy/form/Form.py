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


        D = spdiags(1.0/np.bincount(cell.flatten()), 0, N, N)
        A = D@A.tocsc()
        B = D@B.tocsc()

        P = P.tocsc()
        Q = Q.tocsc()
        S = S.tocsc()

        M = A.transpose()@P@A + A.transpose()@Q@B + B.transpose()@Q.transpose()@A+B.transpose()@S@B 

        P = coo_matrix((N, N), dtype=np.float)
        Q = coo_matrix((N, N), dtype=np.float)
        S = coo_matrix((N, N), dtype=np.float)
        for i in range(2):
            for j in range(2):
                if i == j:
                    val = 1/3
                else:
                    val = 1/6
                P += coo_matrix((self.sigma*val*n[:, 0]*n[:, 0]/h, (bdEdge[:, i], bdEdge[:, j])), shape=(N, N))
                Q += coo_matrix((self.sigma*val*n[:, 0]*n[:, 1]/h, (bdEdge[:, i], bdEdge[:, j])), shape=(N, N))
                S += coo_matrix((self.sigma*val*n[:, 1]*n[:, 1]/h, (bdEdge[:, i], bdEdge[:, j])), shape=(N, N))
          
        P = P.tocsc()
        Q = Q.tocsc()
        S = S.tocsc()

        M += A.transpose()@P@A + A.transpose()@Q@B + B.transpose()@Q@A + B.transpose()@S@B
        return M
