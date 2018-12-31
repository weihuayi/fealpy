
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, block_diag
import numpy as np
from ..quadrature  import TriangleQuadrature
from ..quadrature import IntervalQuadrature
from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace 
from ..boundarycondition import DirichletBC
from ..solver import solve

from .integral_alg import IntegralAlg

class BiharmonicRecoveryFEMModel:
    def __init__(self, mesh, pde, p, q, rtype='simple', dirichlet=True):
        self.space = LagrangeFiniteElementSpace(mesh, p=p) 

        self.uh = self.space.function()
        self.rgh = self.space.function(dim=2)
        self.rlh = self.space.function()

        self.uI = self.space.interpolation(pde.solution) 

        self.pde = pde
        self.integrator = mesh.integrator(q)
        self.rtype = rtype 
        self.dirichlet = dirichlet

        self.gradphi = self.space.mesh.grad_lambda()
        self.measure = mesh.entity_measure('cell')
        self.A, self.B = self.get_revcover_matrix()

        self.integralalg = IntegralAlg(self.integrator, mesh, self.measure)

    def grad_recover_estimate(self):
        qf = self.integrator
        bcs, ws = qf.quadpts, qf.weights
        val0 = self.uh.grad_value(bcs)
        val1 = self.rgh.value(bcs)
        e = np.sum((val1 - val0)**2, axis=-1)
        e = np.einsum('i, ij->j', ws, e)
        e *= self.measure
        e = np.sqrt(e)
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
        e *= self.measure
        e = np.sqrt(e)
        return e 

    def recover_grad(self):
        uh = self.uh
        self.rgh[:, 0] = self.A@uh
        self.rgh[:, 1] = self.B@uh
        mesh = self.space.mesh
        node = mesh.node
        isBdNodes = mesh.ds.boundary_node_flag()
        val = self.pde.gradient(node[isBdNodes])
        isNotNan = np.isnan(val)
        self.rgh[isBdNodes][isNotNan] = val[isNotNan] 

        self.rgh[isBdNodes, :] = self.pde.gradient(node[isBdNodes, :])

    def recover_laplace(self):
        b = np.array([1/3, 1/3, 1/3])
        mesh = self.space.mesh
        N = mesh.number_of_nodes()
        cell = mesh.ds.cell
        val = self.rgh.div_value(b)
        if self.rtype is 'simple':
            np.add.at(self.rlh, cell, val.reshape(-1, 1))
            self.rlh /= np.bincount(cell.flat, minlength=N)
        elif self.rtype is 'harmonic':
            inva = 1/self.measure
            s = np.zeros(N, dtype=np.float)
            np.add.at(s, cell, inva.reshape(-1, 1))
            val *= inva 
            np.add.at(self.rlh, cell, val.reshape(-1, 1))
            self.rlh /= s
        else:
            self.rlh[:] = self.A@self.rgh[:, 0] + self.B@self.rgh[:, 1]
            #raise ValueError("I have not coded the method {}".format(self.rtype))


    def get_revcover_matrix(self):
        measure = self.measure
        gradphi = self.gradphi 

        space = self.space
        mesh = space.mesh

        NC = mesh.number_of_cells() 
        NN = mesh.number_of_nodes() 
        cell = mesh.ds.cell

        if self.rtype is 'simple':
            D = spdiags(1.0/np.bincount(cell.flat), 0, NN, NN)
            I = np.einsum('k, ij->ijk', np.ones(3), cell)
            J = I.swapaxes(-1, -2)
            val = np.einsum('k, ij->ikj', np.ones(3), gradphi[:, :, 0])
            A = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
            val = np.einsum('k, ij->ikj', np.ones(3), gradphi[:, :, 1])
            B = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
        elif self.rtype is 'harmonic':
            gphi = gradphi/measure.reshape(-1, 1, 1)
            d = np.zeros(NN, dtype=np.float)
            np.add.at(d, cell, 1/measure.reshape(-1, 1))
            D = spdiags(1/d, 0, NN, NN)
            I = np.einsum('k, ij->ijk', np.ones(3), cell)
            J = I.swapaxes(-1, -2)
            val = np.einsum('ij, k->ikj',  gphi[:, :, 0], np.ones(3))
            A = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
            val = np.einsum('ij, k->ikj',  gphi[:, :, 1], np.ones(3))
            B = D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN))
        else:
            N = int(np.sqrt(NN))
            h = 2/(N-1)
            idx = np.arange(NN, dtype=np.int).reshape(N, N)

            I = np.r_[idx[1:-1, :].flat, idx[1:-1, :].flat]
            J = np.r_[idx[2:, :].flat, idx[0:-2, :].flat]
            ones = np.ones(NN-2*N, dtype=np.float)
            val = np.r_[ones, -ones]
            A = coo_matrix((val, (I, J)), shape=(NN, NN), dtype=np.float)

            I = np.r_[idx[0, :], idx[0, :], idx[-1, :], idx[-1, :]]
            J = np.r_[idx[0, :], idx[1, :], idx[-1, :], idx[-2, :]]
            val = 2*np.ones(N, dtype=np.float)
            val = np.r_[-val, val, val, -val]
            A += coo_matrix((val, (I, J)), shape=(NN, NN), dtype=np.float)
            A = A.tocsr()/h

            I = np.r_[idx[:, 1:-1].flat, idx[:, 1:-1].flat]
            J = np.r_[idx[:, 2:].flat, idx[:, 0:-2].flat]
            val = np.ones(NN - 2*N, dtype=np.float)
            val = np.r_[val, -val]
            B = coo_matrix((val, (I, J)), shape=(NN, NN), dtype=np.float)

            I = np.r_[idx[:, 0], idx[:, 0], idx[:, -1], idx[:, -1]]
            J = np.r_[idx[:, 0], idx[:, 1], idx[:, -1], idx[:, -2]]
            val = 2*np.ones(N, dtype=np.float)
            val = np.r_[-val, val, val, -val]
            B += coo_matrix((val, (I, J)), shape=(NN, NN), dtype=np.float)
            B = B.tocsr()/h

        return A, B

    def get_left_matrix(self):
        space = self.space

        mesh = space.mesh
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

        gradphi, measure = self.gradphi, self.measure


        I = np.einsum('ij, k->ijk',  cell, np.ones(3))
        J = I.swapaxes(-1, -2)
        val = np.einsum('i, ij, ik->ijk', measure, gradphi[:, :, 0], gradphi[:, :, 0])
        P = csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
        val = np.einsum('i, ij, ik->ijk', measure, gradphi[:, :, 0], gradphi[:, :, 1])
        Q = csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
        val = np.einsum('i, ij, ik->ijk', measure, gradphi[:, :, 1], gradphi[:, :, 1])
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


        # Drichlet term 
        if self.dirichlet:
            I = np.einsum('ij, k->ijk', bdEdge, np.ones(2))
            J = I.swapaxes(-1, -2)
            val = np.array([(1/3, 1/6), (1/6, 1/3)])
            val = np.einsum('i, jk->ijk', 1/h**3, val)
            D = csr_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
            M += D

        return M
    
    def get_laplace_matrix(self):
        space = self.space
        mesh = space.mesh
        N = mesh.number_of_nodes() 
        cell = mesh.ds.cell
        
        gradphi, measure = self.gradphi, self.measure

        I = np.einsum('ij, k->ijk',  cell, np.ones(3))
        J = I.swapaxes(-1, -2)
        val = np.einsum('i, ij, ik->ijk', measure, gradphi[:, :, 0], gradphi[:, :, 0])
        P = csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
        val = np.einsum('i, ij, ik->ijk', measure, gradphi[:, :, 0], gradphi[:, :, 1])
        Q = csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))
        val = np.einsum('i, ij, ik->ijk', measure, gradphi[:, :, 1], gradphi[:, :, 1])
        S = csc_matrix((val.flat, (I.flat, J.flat)), shape=(N, N))

        A, B = self.A, self.B

        M = A.transpose()@P@A + A.transpose()@Q@B + B.transpose()@Q.transpose()@A+B.transpose()@S@B 

        return M

    def get_laplace_dirichlet_vector(self):
        """
        \int_\Gamma g_2 G(\nabla v_h) \cdot n ds
        """
        space = self.space
        mesh = space.mesh
        NC = mesh.number_of_cells() 
        N = mesh.number_of_nodes() 
        cell = mesh.ds.cell
        node = mesh.node

        edge2cell = mesh.ds.edge_to_cell()
        edge = mesh.ds.edge
        isBdEdge = (edge2cell[:,0]==edge2cell[:,1])
        bdEdge = edge[isBdEdge]

        b0 = np.zeros(N, dtype=np.float)
        b1 = np.zeros(N, dtype=np.float)

        qf = IntervalQuadrature(5)
        bcs, ws = qf.quadpts, qf.weights

        pp = np.einsum('...j, ijk->...ik', bcs, node[bdEdge])
        val = self.pde.laplace_dirichlet(pp)#TODO: modify pde

        val0 = np.einsum('m, mj, i, mi->ij', ws, bcs, n[:, 0], val)
        np.add.at(b0, bdEdge, val0)
        
        val1 = np.einsum('m, mj, i, mi->ij', ws, bcs, n[:, 1], val)
        np.add.at(b1, bdEdge, val1)

        return self.A.transpose()@b0 + self.B.transpose()@b1 #TODO: check it 

    def get_laplace_neu_vector(self):
        """
        \int_\Gamma g_2 G(\nabla v_h) \cdot n ds
        """
        space = self.space
        mesh = space.mesh
        NC = mesh.number_of_cells() 
        N = mesh.number_of_nodes() 
        cell = mesh.ds.cell
        node = mesh.node

        edge2cell = mesh.ds.edge_to_cell()
        edge = mesh.ds.edge
        isBdEdge = (edge2cell[:,0]==edge2cell[:,1])
        bdEdge = edge[isBdEdge]



    def get_right_vector(self):
        space = self.space
        mesh = space.mesh
        pde = self.pde
        
        qf = self.integrator 
        bcs, ws = qf.quadpts, qf.weights
        pp = mesh.bc_to_point(bcs)
        fval = pde.source(pp)
        phi = space.basis(bcs)
        bb = np.einsum('i, ij, ik->kj', ws, phi,fval)

        bb *= self.measure.reshape(-1, 1)
        gdof = space.number_of_global_dofs()
        cell2dof = space.dof.cell2dof        
        b = np.zeros((gdof,), dtype=np.float)
        np.add.at(b, cell2dof, bb)
        if self.dirichlet:
            return b + self.get_neuman_vector() + self.get_dirichlet_vector()
        else:
            return b + self.get_neuman_vector()

    def get_neuman_vector(self):
        space = self.space
        mesh = space.mesh
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
        val = self.pde.neuman(pp, n)

        val0 = np.einsum('m, mj, i, mi->ij', ws, bcs, n[:, 0]/h, val)
        np.add.at(b0, bdEdge, val0)
        
        val1 = np.einsum('m, mj, i, mi->ij', ws, bcs, n[:, 1]/h, val)
        np.add.at(b1, bdEdge, val1)

        return self.A.transpose()@b0 + self.B.transpose()@b1

    def get_dirichlet_vector(self):
        space = self.space
        mesh = space.mesh
        cell = mesh.ds.cell
        node = mesh.node

        NN = mesh.number_of_nodes()

        edge = mesh.ds.edge
        isBdEdge = mesh.ds.boundary_edge_flag()
        bdEdge = edge[isBdEdge]

        n = node[bdEdge[:, 1],] - node[bdEdge[:, 0],:]
        h = np.sqrt(np.sum(n**2, axis=1)) 

        b = np.zeros(NN, dtype=np.float)

        qf = IntervalQuadrature(5)
        bcs, ws = qf.quadpts, qf.weights
        pp = np.einsum('...j, ijk->...ik', bcs, node[bdEdge])
        val = self.pde.dirichlet(pp)
        val = np.einsum('m, mj, mi->ij', ws, bcs, val/h**3)
        np.add.at(b, bdEdge, val)

        return b

    def solve(self):
        if self.dirichlet:
            solve(self, self.uh, dirichlet=None, solver='direct')
        else:
            bc = DirichletBC(self.space, self.pde.dirichlet)
            solve(self, self.uh, dirichlet=bc, solver='direct')
        self.recover_grad()
        self.recover_laplace()


    def get_error(self):
        u = self.pde.solution
        uh = self.uh.value
        e0 = self.integralalg.L2_error(u, uh)

        gu = self.pde.gradient
        guh = self.uh.grad_value
        e1 = self.integralalg.L2_error(gu, guh)

        rguh = self.rgh.value
        e2 = self.integralalg.L2_error(gu, rguh)

        lu = self.pde.laplace
        luh = self.rgh.div_value
        e3 = self.integralalg.L2_error(lu, luh)

        rluh = self.rlh.value
        e4 = self.integralalg.L2_error(lu, rluh)
        
        return e0, e1, e2, e3, e4
