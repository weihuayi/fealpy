import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from ..quadrature import IntervalQuadrature
from ..functionspace.barycentric_coordinates import bc_to_point, grad_lambda


class DirichletBC:
    def __init__(self, V, g0, is_boundary_dof=None, dtype=np.float):
        self.V = V
        self.g0 = g0

        gdof = V.number_of_global_dofs()
        if is_boundary_dof == None:
            isBdDof = np.zeros(gdof, dtype=np.bool)
            edge2dof = V.edge_to_dof()
            isBdEdge = V.mesh.ds.boundary_edge_flag()
            isBdDof[edge2dof[isBdEdge]] = True
        else:
            ipoints = V.interpolation_points()
            isBdDof = is_boundary_dof(ipoints)

        self.isBdDof = isBdDof
        self.dtype = dtype

    def apply(self, A, b):
        """ Modify matrix A and b
        """
        g0 = self.g0
        V = self.V
        isBdDof = self.isBdDof

        gdof = V.number_of_global_dofs()

        x = np.zeros((gdof,), dtype=self.dtype)

        ipoints = V.interpolation_points()
        x[isBdDof] = g0(ipoints[isBdDof,:])
        b -= A@x

        bdIdx = np.zeros(gdof, dtype=np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, gdof, gdof)
        T = spdiags(1-bdIdx, 0, gdof, gdof)
        A = T@A@T + Tbd

        b[isBdDof] = x[isBdDof] 
        return A, b

    def apply_on_matrix(self, A):

        V = self.V
        isBdDof = self.isBdDof
        gdof = V.number_of_global_dofs()

        bdIdx = np.zeros((A.shape[0], ), np.int)
        bdIdx[isBdDof] = 1
        Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd

        return A

    def apply_on_vector(self, b, A):
        
        g0 = self.g0
        V = self.V
        isBdDof = self.isBdDof

        gdof = V.number_of_global_dofs()
        x = np.zeros((gdof,), dtype=self.dtype)

        x[isBdDof] = g0(ipoints[isBdDof,:])
        b -= A@x

        b[isBdDof] = x[isBdDof] 

        return b

class BihamonicRecoveryBC():
    def __init__(self, V, g, sigma=1, dtype=np.float):
        self.V = V
        self.g = g
        self.sigma = sigma
        self.dtype = dtype

    def get_vector(self):
        V = self.V
        mesh = V.mesh
        cell = mesh.ds.cell
        point = mesh.point

        edge = mesh.ds.edge
        isBdEdge = mesh.ds.boundary_edge_flag()
        bdEdge = edge[isBdEdge]

        edge2cell = mesh.ds.edge2cell
        bdCellIndex = edge2cell[isBdEdge, 0]
        # find all boundary cell 
        bdCell = cell[bdCellIndex, :]

        NC = bdCell.shape[0] # the number of boundary cells

        # the unit outward normal on boundary edge
        W = np.array([[0, -1], [1, 0]], dtype=np.int)
        n = (point[bdEdge[:,1],] - point[bdEdge[:,0],:])@W
        h = np.sqrt(np.sum(n**2, axis=1)) 
        n /= h.reshape((-1,1))

        ldof = V.number_of_local_dofs() 
        gdof = V.number_of_global_dofs()

        bb = np.zeros((NC, ldof), dtype=self.dtype)
        qf = IntervalQuadrature(5)
        nQuad = qf.get_number_of_quad_points()
        gradphi, _ = grad_lambda(point, bdCell)
        for i in range(nQuad):
            lambda_k, w_k = qf.get_gauss_point_and_weight(i)
            p = point[bdEdge[:, 0], :]*lambda_k[0] \
                    + point[bdEdge[:, 1], :]*lambda_k[1] 
            val = self.g(p, n)
            for j in range(ldof):
                bb[:, j] += np.sum(gradphi[:, j, :]*n, axis=1)*val*w_k

        bb /= (h.reshape(-1, 1))

        b = np.zeros((gdof,), dtype=self.dtype)
        np.add.at(b, bdCell.flatten(), bb.flatten())
        return self.sigma*b


class BihamonicRecoveryBC1():
    def __init__(self, V, g, sigma=1, dtype=np.float):
        self.V = V
        self.g = g
        self.sigma = sigma
        self.dtype = dtype

    def get_vector(self):
        V = self.V
        mesh = V.mesh
        cell = mesh.ds.cell
        point = mesh.point

        N = mesh.number_of_points()

        edge = mesh.ds.edge
        isBdEdge = mesh.ds.boundary_edge_flag()
        bdEdge = edge[isBdEdge]

        A = coo_matrix((N, N), dtype=np.float)
        B = coo_matrix((N, N), dtype=np.float)
        gradphi, area = mesh.grad_lambda()
        for i in range(3):
            for j in range(3):  
                A += coo_matrix((gradphi[:,j,0]/area, (cell[:,i], cell[:,j])), shape=(N,N))
                B += coo_matrix((gradphi[:,j,1]/area, (cell[:,i], cell[:,j])), shape=(N,N))

                #A += coo_matrix((gradphi[:,j,0], (cell[:,i], cell[:,j])), shape=(N,N))
                #B += coo_matrix((gradphi[:,j,1], (cell[:,i], cell[:,j])), shape=(N,N))

        d = np.bincount(cell[:, 0], weights=1/area, minlength=N)
        d += np.bincount(cell[:, 1], weights=1/area, minlength=N)
        d += np.bincount(cell[:, 2], weights=1/area, minlength=N)

        D = spdiags(d, 0, N, N)
        #D = spdiags(1.0/np.bincount(cell.flatten()), 0, N, N)
        A = D@A.tocsc()
        B = D@B.tocsc()

        # the unit outward normal on boundary edge
        W = np.array([[0, -1], [1, 0]], dtype=np.int)
        n = (point[bdEdge[:,1],] - point[bdEdge[:,0],:])@W
        h = np.sqrt(np.sum(n**2, axis=1)) 
        n /= h.reshape((-1,1))


        b0 = np.zeros(N, dtype=self.dtype)
        b1 = np.zeros(N, dtype=self.dtype)

        qf = IntervalQuadrature(5)
        nQuad = qf.get_number_of_quad_points()
        for i in range(nQuad):
            lambda_k, w_k = qf.get_gauss_point_and_weight(i)
            p = point[bdEdge[:, 0], :]*lambda_k[0] \
                    + point[bdEdge[:, 1], :]*lambda_k[1] 
            val = self.g(p, n)
            b0[bdEdge[:, 0]] += w_k*n[:, 0]*val*lambda_k[0]/h
            b0[bdEdge[:, 1]] += w_k*n[:, 0]*val*lambda_k[1]/h
            b1[bdEdge[:, 0]] += w_k*n[:, 1]*val*lambda_k[0]/h
            b1[bdEdge[:, 1]] += w_k*n[:, 1]*val*lambda_k[1]/h

        return self.sigma*(A.transpose()@b0 + B.transpose()@b1)
        


