import numpy as np
from scipy.sparse import coo_matrix, bmat

from ..functionspace import WeakGalerkinSpace2d

class SobolevEquationWGModel2d:
    def __init__(self, pde, mesh, p, q=None):
        self.pde = pde
        self.mesh = mesh
        self.space = WeakGalerkinSpace2d(mesh, p=p, q=q)
        self.construct_marix()

    def init_solution(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        uh = self.space.function(dim=NL)
        return uh

    def construct_marix(self):
        """
        构造 Soblove 方程对应的所有矩阵
        """
        gdof = self.space.number_of_global_dofs()
        cell2dof, cell2dofLocation = self.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        R0 = np.hsplit(self.space.R0, cell2dofLocation[1:-1])
        R1 = np.hsplit(self.space.R1, cell2dofLocation[1:-1])

        H0 = self.space.H0

        f0 = lambda x: x[0].T@x[1]@x[2]
        M00 = list(map(f0, zip(R0, H0, R0)))
        M01 = list(map(f0, zip(R0, H0, R1)))
        M11 = list(map(f0, zip(R1, H0, R1)))

        f1 = lambda x: np.repeat(x, x.shape[0])
        f2 = lambda x: np.tile(x, x.shape[0])
        f3 = lambda x: x.flat

        I = np.concatenate(list(map(f1, cd)))
        J = np.concatenate(list(map(f2, cd)))

        val = np.concatenate(list(map(f3, M00)))
        M00 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        val = np.concatenate(list(map(f3, M01)))
        M01 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        val = np.concatenate(list(map(f3, M11)))
        M11 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        self.D = bmat([[M00, M01], [M01.T, M11]], format='csr') # weak divergence matrix
        self.G = M00 + M11 # weak gradient matrix


        cell2dof = self.space.cell_to_dof(doftype='cell') # only get the dofs in cell
        ldof = cell2dof.shape[1]
        gdof = self.space.number_of_global_dofs()

        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        M = csr_matrix(
                (self.space.CM.flat, (I.flat, J.flat)), shape=(gdof, gdof)
                )
        self.M = bmat([[M, csr_matrix((gdof, gdof))], [csr_matrix((gdof, gdof)), M]], format='csr')

        self.A1 = self.D + self.M/self.pde.mu
        self.A2 = self.G*self.pdf.mu

    def construct_s_matrix(self):
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        qf = self.space.integralalg.edgeintegrator
        bcs, ws = qf.quadpts, qf.weights

        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])

        phi0 = self.smspace.basis(ps, cellidx=edge2cell[:, 0])
        phi1 = self.smspace.basis(
                ps[:, isInEdge, :],
                cellidx=edge2cell[isInEdge, 1]
                )
        phi = self.edge_basis(bcs)

        edge2dof = self.space.dof.edge_to_dof()
        cell2dof = self.space.cell_to_dof(doftype='cell')

        F0 = -np.einsum('i, ijm, ijn->mjn', ws, phi0, phi)
        F1 = -np.einsum('i, ijm, ijn->mjn', ws, phi1, phi[:, isInEdge, :])

        F2 = -np.einsum('i, ijm, ijn->mjn', ws, phi, phi)

    def solution_projection(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        uh = self.space.function(dim=NL)
        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            uh[:, i] = self.space.projection(lambda x:self.pde.solution(x, t))
        return uh

    def source_projection(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        fh = self.space.function(dim=NL)
        times = timeline.all_time_levels()
        for i, t in enumerate(times):
            fh[:, i] = self.space.projection(lambda x:self.pde.source(x, t))
        return fh

    def get_current_left_matrix(self, timeline):
        return self.G

    def get_current_right_vector(self, data, timeline):
        mu = self.pde.mu
        epsilon = self.pde.epsilon

        uh = data[0]
        ph = data[1]
        fh = data[2]
        solver = data[4]
        i = timeline.current

        cell2dof, cell2dofLocation = self.cell_to_dof(doftype='all')
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        R0 = self.space.R0
        R1 = self.space.R1

        cell2dof = self.space.cell_to_dof(doftype='cell') # only get the dofs in cell
        # epsilon/mu(\nabla u, \bfq)
        F0 = np.zeros((gdof, 2), dtype=self.space.ftype)
        f00 = lambda j: R0[:, cell2dofLocation[j]:cell2dofLocation[j+1]]@uh[cd[j], i]
        f01 = lambda j: R1[:, cell2dofLocation[j]:cell2dofLocation[j+1]]@uh[cd[j], i]
        F0[cell2dof.flat, 0] = np.concatenate(list(map(f00, range(NC))))
        F0[cell2dof.flat, 1] = np.concatenate(list(map(f01, range(NC))))

        # -(f, \nabla\cdot \bfq)
        F1 = np.zeros((gdof, 2), dtype=self.space.ftype)
        f11 = lambda j: fh[cell2dof[j, :], i]@R0[:, cell2dofLocation[j]:cell2dofLocation[j+1]]
        f12 = lambda j: fh[cell2dof[j, :], i]@R1[:, cell2dofLocation[j]:cell2dofLocation[j+1]]
        F1[cell2dof.flat, 0] = np.concatenate(
            list(map(f11, range(NC)))
            )
        F1[cell2dof.flat, 1] = np.concatenate(
            list(map(f12, range(NC)))
            )

        F = epsilon/mu*F0 - F1
        ph[:, i] = solver(self.A1, F.reshape(-1, order='F'))

        dt = timeline.current_time_step_length()
        t0 = timeline.current_time_level()
        f0 = lambda x: self.pde.source(x, t0) + self.pde.source(x, t1)

    def apply_boundary_condition(self, A, b, timeline):
        t1 = timeline.next_time_level()
        bc = DirichletBC(self.space, lambda x:self.pde.dirichlet(x, t1))
        A, b = bc.apply(A, b)
        return A, b
