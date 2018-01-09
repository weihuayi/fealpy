
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye
import numpy as np
from ..quadrature  import TriangleQuadrature
from ..quadrature import IntervalQuadrature
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace 
from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace 

class BiharmonicRecoveryFEMModel:
    def __init__(self, mesh, model, integrator=None, rtype='simple'):
        self.V = LagrangeFiniteElementSpace(mesh, p=1) 
        self.VV = VectorLagrangeFiniteElementSpace(mesh, p=1)

        self.uh = self.V.function()
        self.uI = self.V.interpolation(model.solution) 
        self.rgh = self.VV.function()
        self.rlh = self.V.function()

        self.model = model
        if integrator is None:
            self.integrator = TriangleQuadrature(p+1)
        if type(integrator) is int:
            self.integrator = TriangleQuadrature(integrator)
        else:
            self.integrator = integrator 
        self.rtype = rtype 

        self.gradphi = V.mesh.grad_lambda()
        self.area = mesh.area()
        self.A, self.B = self.get_revcover_matrix()

    def grad_recover_estimate(self):
        qf = self.integrator
        bcs, ws = qf.quadpts, qf.weights
        val0 = self.uh.grad_value(bcs)
        val1 = self.ruh.value(bcs)
        e = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, e)
        e *= self.area
        return e 

    def laplace_recover_estimate(self, etype=1):
        qf = self.integrator
        bcs, ws = qf.quadpts, qf.weights
        if etype == 1:
            val0 = self.rgh.div_value(bcs)
            val1 = self.rlh.value(bcs)
            e = np.einsum('i, ij->j', ws, (val1 - val0)**2)
        elif etype == 2:
            val0 = self.rgh.div_value(bcs)
            e = np.einsum('i, ij->j', ws, val0**2)
        elif etype == 3:
            val1 = self.rlh.value(bcs)
            e = np.einsum('i, ij->j', ws, val1**2)
        else:
            raise ValueError("1 <= etype <=3! Your input is {}".format(etype)) 
        e *= self.area
        e = np.sqrt(e)
        return e 

    def recover_grad(self):
        uh = self.uh
        self.rgh[:, 0] = self.A@uh
        self.rgh[:, 1] = self.B@uh
        mesh = self.V.mesh
        point = mesh.point
        isBdPoints = mesh.ds.boundary_point_flag()
        val = self.model.gradient(point[isBdPoints])
        isNotNan = np.isnan(val)
        self.rgh[isBdPoints][isNotNan] = val[isNotNan] 

    def recover_laplace(self):
        b = np.array([1/3, 1/3, 1/3])
        mesh = self.V.mesh
        N = mesh.number_of_points()
        cell = mesh.ds.cell
        val = self.rgh.div_value(b)
        if self.rtype is 'simple':
            np.add.at(self.rlh, cell, val.reshape(-1, 1))
            self.rlh /= np.bincount(cell.flat, minlength=N)
        elif self.rtype is 'harmonic':
            inva = 1/self.area
            s = np.zeros(N, dtype=np.float)
            np.add.at(s, cell, inva.reshape(-1, 1))
            val *= inva 
            np.add.at(self.rlh, cell, val.reshape(-1, 1))
            self.rlh /= s
        else:
            raise ValueError("I have not coded the method {}".format(self.rtype))


    def get_revcover_matrix(self):
        area = self.area
        gradphi = self.gradphi 

        V = self.V
        mesh = V.mesh

        NC = mesh.number_of_cells() 
        N = mesh.number_of_points() 
        cell = mesh.ds.cell

        A = coo_matrix((N, N), dtype=np.float)
        B = coo_matrix((N, N), dtype=np.float)
        if self.rtype is 'simple':
            for i in range(3):
                for j in range(3):  
                    A += coo_matrix((gradphi[:,j,0], (cell[:,i], cell[:,j])), shape=(N,N))
                    B += coo_matrix((gradphi[:,j,1], (cell[:,i], cell[:,j])), shape=(N,N))
            D = spdiags(1.0/np.bincount(cell.flat), 0, N, N)
            A = D@A.tocsc()
            B = D@B.tocsc()
        elif self.rtype is 'inv_area':
            for i in range(3):
                for j in range(3):  
                    A += coo_matrix((gradphi[:,j,0]/area, (cell[:,i], cell[:,j])), shape=(N,N))
                    B += coo_matrix((gradphi[:,j,1]/area, (cell[:,i], cell[:,j])), shape=(N,N))
            d = np.bincount(cell[:, 0], weights=1/area, minlength=N)
            d += np.bincount(cell[:, 1], weights=1/area, minlength=N)
            d += np.bincount(cell[:, 2], weights=1/area, minlength=N)

            D = spdiags(1/d, 0, N, N)
            A = D@A.tocsc()
            B = D@B.tocsc()
        else:
            raise ValueError("I have not coded the method {}".format(self.rtype))

        return A, B

    def get_left_matrix(self):
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

        P = coo_matrix((N, N), dtype=np.float)
        Q = coo_matrix((N, N), dtype=np.float)
        S = coo_matrix((N, N), dtype=np.float)
        gradphi, area = self.gradphi, self.area
        for i in range(3):
            for j in range(3):  
                val00 = gradphi[:,i,0]*gradphi[:,j,0]*area 
                val01 = gradphi[:,i,0]*gradphi[:,j,1]*area
                val11 = gradphi[:,i,1]*gradphi[:,j,1]*area
                P += coo_matrix((val00, (cell[:,i], cell[:,j])), shape=(N, N))
                Q += coo_matrix((val01, (cell[:,i], cell[:,j])), shape=(N, N))
                S += coo_matrix((val11, (cell[:,i], cell[:,j])), shape=(N, N))
        P = P.tocsc()
        Q = Q.tocsc()
        S = S.tocsc()

        A, B = self.A, self.B

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
                P += coo_matrix((val*n[:, 0]*n[:, 0]/h, (bdEdge[:, i], bdEdge[:, j])), shape=(N, N))
                Q += coo_matrix((val*n[:, 0]*n[:, 1]/h, (bdEdge[:, i], bdEdge[:, j])), shape=(N, N))
                S += coo_matrix((val*n[:, 1]*n[:, 1]/h, (bdEdge[:, i], bdEdge[:, j])), shape=(N, N))
          
        P = P.tocsc()
        Q = Q.tocsc()
        S = S.tocsc()

        M += A.transpose()@P@A + A.transpose()@Q@B + B.transpose()@Q@A + B.transpose()@S@B
        return M

    def get_right_vector(self):
        V = self.V
        mesh = V.mesh

        NC = mesh.number_of_cells()
        qf = TriangleQuadrature(5)
        nQuad = qf.get_number_of_quad_points()

        gdof = V.number_of_global_dofs()
        ldof = V.number_of_local_dofs()
        cell2dof = V.cell_to_dof()
        
        bb = np.zeros((NC, ldof), dtype=np.float)
        area = mesh.area()
        for i in range(nQuad):
            lambda_k, w_k = qf.get_gauss_point_and_weight(i)
            p = mesh.bc_to_point(lambda_k)
            fval = self.model.source(p)
            phi = V.basis(lambda_k)
            for j in range(ldof):
                bb[:, j] += fval*phi[j]*w_k

        bb *= area.reshape(-1, 1)
        cell2dof = V.cell_to_dof()
        b = np.zeros((gdof,), dtype=np.float)
        np.add.at(b, cell2dof, bb)
        return b + self.get_neuman_vector()

    def get_neuman_vector(self):
        V = self.V
        mesh = V.mesh
        cell = mesh.ds.cell
        point = mesh.point

        N = mesh.number_of_points()

        edge = mesh.ds.edge
        isBdEdge = mesh.ds.boundary_edge_flag()
        bdEdge = edge[isBdEdge]

        # the unit outward normal on boundary edge
        W = np.array([[0, -1], [1, 0]], dtype=np.int)
        n = (point[bdEdge[:,1],] - point[bdEdge[:,0],:])@W
        h = np.sqrt(np.sum(n**2, axis=1)) 
        n /= h.reshape((-1,1))


        b0 = np.zeros(N, dtype=np.float)
        b1 = np.zeros(N, dtype=np.float)

        qf = IntervalQuadrature(5)
        nQuad = qf.get_number_of_quad_points()
        for i in range(nQuad):
            lambda_k, w_k = qf.get_gauss_point_and_weight(i)
            p = point[bdEdge[:, 0], :]*lambda_k[0] \
                    + point[bdEdge[:, 1], :]*lambda_k[1] 
            val = self.model.neuman(p, n)
            b0[bdEdge[:, 0]] += w_k*n[:, 0]*val*lambda_k[0]/h
            b0[bdEdge[:, 1]] += w_k*n[:, 0]*val*lambda_k[1]/h
            b1[bdEdge[:, 0]] += w_k*n[:, 1]*val*lambda_k[0]/h
            b1[bdEdge[:, 1]] += w_k*n[:, 1]*val*lambda_k[1]/h

        return self.A.transpose()@b0 + self.B.transpose()@b1

    def solve(self):
        bc = DirichletBC(self.V, self.model.dirichlet)
        solve(self, self.uh, dirichlet=bc, solver='direct')
