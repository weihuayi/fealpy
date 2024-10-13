
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye
import numpy as np
from ..quadrature  import TriangleQuadrature
from ..quadrature import IntervalQuadrature
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace 
from ..functionspace.lagrange_fem_space import VectorLagrangeFiniteElementSpace 
from ..functionspace.lagrange_fem_space import SymmetricTensorLagrangeFiniteElementSpace 
from ..boundarycondition import DirichletBC
from ..solver import solve
from .integral_alg import IntegralAlg


class BiharmonicRecoveryFEMModel:
    def __init__(self, mesh, model, integrator=None, rtype='simple'):
        self.V = LagrangeFiniteElementSpace(mesh, p=1) 
        self.V2 = VectorLagrangeFiniteElementSpace(mesh, p=1)
        self.V3 = SymmetricTensorLagrangeFiniteElementSpace(mesh, 1) 

        self.uh = self.V.function()
        self.uI = self.V.interpolation(model.solution) 
        self.rgh = self.V2.function()
        self.rlh = self.V.function()

        self.model = model
        if integrator is None:
            self.integrator = TriangleQuadrature(p+1)
        if type(integrator) is int:
            self.integrator = TriangleQuadrature(integrator)
        else:
            self.integrator = integrator 
        self.rtype = rtype 

        self.gradphi = self.V.mesh.grad_lambda()
        self.area = mesh.area()
        self.A, self.B = self.get_revcover_matrix()


    def reinit(self, mesh):
        self.V = LagrangeFiniteElementSpace(mesh, p=1) 
        self.V2 = VectorLagrangeFiniteElementSpace(mesh, p=1)
        self.uh = self.V.function()
        self.uI = self.V.interpolation(self.model.solution)
        self.rgh = self.V2.function()
        self.rlh = self.V.function()
        self.area = self.V.mesh.area()
        self.gradphi = self.V.mesh.grad_lambda()
        self.A, self.B = self.get_revcover_matrix()

    def grad_recover_estimate(self):
        qf = self.integrator
        bcs, ws = qf.quadpts, qf.weights
        val0 = self.uh.grad_value(bcs)
        val1 = self.rgh.value(bcs)
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
        self.rgh[:,0] = self.A@uh
        self.rgh[:,1] = self.B@uh
        mesh = self.V.mesh
        node = mesh.node
        isBdNodes = mesh.ds.boundary_node_flag()
        val = self.model.gradient(node[isBdNodes])
        isNotNan = np.isnan(val)
        self.rgh[isBdNodes][isNotNan] = val[isNotNan] 

    def recover_laplace(self):
        b = np.array([1/3, 1/3, 1/3])
        mesh = self.V.mesh
        N = mesh.number_of_nodes()
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
        N = mesh.number_of_nodes() 
        cell = mesh.ds.cell

        if self.rtype is 'simple':
            D = spdiags(1.0/np.bincount(cell.flat), 0, N, N)
            I = np.einsum('k, ij->ijk', np.ones(3), cell)
            J = I.swapaxes(-1, -2)
            val = np.einsum('k, ij->ikj', np.ones(3), gradphi[:, :, 0])
            A = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
            val = np.einsum('k, ij->ikj', np.ones(3), gradphi[:, :, 1])
            B = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
        elif self.rtype is 'harmonic':
            gphi = gradphi/area.reshape(-1, 1, 1)
            d = np.zeros(N, dtype=np.float)
            np.add.at(d, cell, 1/area.reshape(-1, 1))
            D = spdiags(1/d, 0, N, N)
            I = np.einsum('k, ij->ijk', np.ones(3), cell)
            J = I.swapaxes(-1, -2)
            val = np.einsum('ij, k->ikj',  gphi[:, :, 0], np.ones(3))
            A = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
            val = np.einsum('ij, k->ikj',  gphi[:, :, 1], np.ones(3))
            B = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
        else:
            raise ValueError("I have not coded the method {}".format(self.rtype))

        return A, B

    def get_left_matrix(self):
        V = self.V

        mesh = V.mesh
        NC = mesh.number_of_cells() 
        N = mesh.number_of_nodes() 
        cell = mesh.ds.cell
        node = mesh.node

        edge2cell = mesh.ds.edge_to_cell()
        edge = mesh.ds.edge
        isBdEdge = (edge2cell[:,0]==edge2cell[:,1])
        bdEdge = edge[isBdEdge]
        
        W = np.array([[0, -1], [1, 0]], dtype=np.int)
        n = (node[bdEdge[:,1],] - node[bdEdge[:,0],:])@W
        h = np.sqrt(np.sum(n**2, axis=1)) 
        n /= h.reshape(-1, 1)

        gradphi, area = self.gradphi, self.area


        I = np.einsum('ij, k->ijk',  cell, np.ones(3))
        J = I.swapaxes(-1, -2)
        val = np.einsum('i, ij, ik->ijk', area, gradphi[:, :, 0], gradphi[:, :, 0])
        P = csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
        val = np.einsum('i, ij, ik->ijk', area, gradphi[:, :, 0], gradphi[:, :, 1])
        Q = csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
        val = np.einsum('i, ij, ik->ijk', area, gradphi[:, :, 1], gradphi[:, :, 1])
        S = csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))

        

        A, B = self.A, self.B

        M = A.transpose()@P@A + A.transpose()@Q@B + B.transpose()@Q.transpose()@A+B.transpose()@S@B 

        I = np.einsum('ij, k->ijk', bdEdge, np.ones(2))
        J = I.swapaxes(-1, -2)
        val = np.array([(1/3, 1/6), (1/6, 1/3)])
        val0 = np.einsum('i, jk->ijk', n[:, 0]*n[:, 0]/h, val)
        P = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(N, N))
        val0 = np.einsum('i, jk->ijk', n[:, 0]*n[:, 1]/h, val)
        Q = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(N, N))
        val0 = np.einsum('i, jk->ijk', n[:, 1]*n[:, 1]/h, val)
        S = csc_matrix((val0.flat, (I.flat, J.flat)), shape=(N, N))


        M += A.transpose()@P@A + A.transpose()@Q@B + B.transpose()@Q@A + B.transpose()@S@B
        return M

    def get_right_vector(self):
        V = self.V
        mesh = V.mesh
        model = self.model
        
        qf = self.integrator 
        bcs, ws = qf.quadpts, qf.weights
        pp = mesh.bc_to_point(bcs)
        fval = model.source(pp)
        phi = V.basis(bcs)
        bb = np.einsum('i, ij, ik->kj', ws, phi,fval)

        bb *= self.area.reshape(-1, 1)
        gdof = V.number_of_global_dofs()
        cell2dof = V.dof.cell2dof        
        b = np.zeros((gdof,), dtype=np.float)
        np.add.at(b, cell2dof, bb)
        return b + self.get_neuman_vector()

    def get_neuman_vector(self):
        V = self.V
        mesh = V.mesh
        cell = mesh.ds.cell
        node = mesh.node

        N = mesh.number_of_nodes()

        edge = mesh.ds.edge
        isBdEdge = mesh.ds.boundary_edge_flag()
        bdEdge = edge[isBdEdge]

        # the unit outward normal on boundary edge
        W = np.array([[0, -1], [1, 0]], dtype=np.int)
        n = (node[bdEdge[:,1],] - node[bdEdge[:,0],:])@W
        h = np.sqrt(np.sum(n**2, axis=1)) 
        n /= h.reshape((-1,1))


        b0 = np.zeros(N, dtype=np.float)
        b1 = np.zeros(N, dtype=np.float)

        qf = IntervalQuadrature(5)
        bcs, ws = qf.quadpts, qf.weights
        pp = np.einsum('...j, ijk->...ik', bcs, node[bdEdge])
        val = self.model.neuman(pp, n)
        val0 = np.einsum('m, mj, i, mi->ij', ws, bcs, n[:, 0]/h, val)
        np.add.at(b0, bdEdge, val0)
        val0 = np.einsum('m, mj, i, mi->ij', ws, bcs, n[:, 1]/h, val)
        np.add.at(b0, bdEdge, val0)

        return self.A.transpose()@b0 + self.B.transpose()@b1
    
    
    def error(self):
        u = self.model.solution
        uh = self.uh.value
        e0 = self.integralalg.L2_error(u, uh)

        gu = self.model.gradient
        guh = self.uh.grad_value
        e1 = self.integralalg.L2_error(gu, guh)
        return e0, e1


    def solve(self):
        bc = DirichletBC(self.V, self.model.dirichlet)
        solve(self, self.uh, dirichlet=bc, solver='direct')
