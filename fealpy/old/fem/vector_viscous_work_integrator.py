import numpy as np

class VectorViscousWorkIntegrator:
    def __init__(self, mu=None, q=None):
        self.mu = mu
        self.q = q 

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        construct the mu * (epslion(u), epslion(v)) fem matrix
        epsion(u) = 1/2*(\nabla u+ (\nabla u).T)
        """

        mesh = space[0].mesh
        ldof = space[0].number_of_local_dofs()
        p = space[0].p
        GD = mesh.geo_dimension()
        q = self.q if self.q is not None else p+1


        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        if GD == 2:
            idx = [(0, 0), (0, 1),  (1, 1)]
            imap = {(0, 0):0, (0, 1):1, (1, 1):2}
        elif GD == 3:
            idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            imap = {(0, 0):0, (0, 1):1, (0, 2):2, (1, 1):3, (1, 2):4, (2, 2):5}

        A = []

        qf =  mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad = space[0].grad_basis(bcs, index=index) # (NQ, NC, ldof, GD)

        NC = len(cellmeasure)

        if out is None:
            K = np.zeros((NC, GD*ldof, GD*ldof), dtype=np.float64)
        else:
            assert out.shape == (NC, GD*ldof, GD*ldof)
            K = out

        A = [np.einsum('i, ijm, ijn, j->jmn', ws, grad[..., i], grad[..., j], cellmeasure, optimize=True) for i, j in idx]

        D = 0
        for i in range(GD):
            D += 1/2*A[imap[(i, i)]]
        
        if space[0].doforder == 'sdofs': # 标量自由度优先排序 
            for i in range(GD):
                for j in range(i, GD):
                    if i == j:
                        K[:, i*ldof:(i+1)*ldof, i*ldof:(i+1)*ldof] += D  
                        K[:, i*ldof:(i+1)*ldof, i*ldof:(i+1)*ldof] += 1/2*A[imap[(i, i)]]
                    else:
                        K[:, i*ldof:(i+1)*ldof, j*ldof:(j+1)*ldof] += 1/2*A[imap[(i, j)]].transpose(0, 2, 1)
                        K[:, j*ldof:(j+1)*ldof, i*ldof:(i+1)*ldof] += 1/2*A[imap[(i, j)]]
        elif space[0].doforder == 'vdims':
            for i in range(GD):
                for j in range(i, GD):
                    if i == j:
                        K[:, i::GD, i::GD] += D 
                        K[:, i::GD, i::GD] += 1/2*A[imap[(i, i)]]
                    else:
                        K[:, i::GD, j::GD] += 1/2*A[imap[(i, j)]].transpose(0, 2, 1)
                        K[:, j::GD, i::GD] += 1/2*A[imap[(i, j)]]
        if out is None:
            return K


    def assembly_cell_matrix_fast(self, space, index=np.s_[:], cellmeasure=None):
        pass


    def assembly_cell_matrix_ref(self, space, index=np.s_[:], cellmeasure=None):
        pass
