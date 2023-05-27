import numpy as np

class PressIntegrator:
    def __init__(self, q=None):
        self.q = q 

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        construct the (pI, nabla v) fem matrix
        """

        mesh = space[0].mesh
        ldof = space[0].number_of_local_dofs()
        p = space[0].p
        GD = mesh.geo_dimension()
        q = self.q if self.q is not None else p+1


        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        qf =  mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gradphi = space[0].grad_basis(bcs, index=index) # (NQ, NC, ldof, GD)
        phi = space[0].basis(bcs, index=index) # (NQ, NC, ldof)

        NC = len(cellmeasure)
        
        if out is None:
            K = np.zeros((NC, GD*ldof, GD*ldof), dtype=np.float64)
        else:
            assert out.shape == (NC, GD*ldof, GD*ldof)
            K = out
        
        if space[0].doforder == 'sdofs': # 标量自由度优先排序 
            for i in range(GD):
                val = np.einsum('i, ijm, ijn, j->jmn', ws, phi, gradphi[..., i], cellmeasure, optimize=True)
                K[:, i*ldof:(i+1)*ldof, i*ldof:(i+1)*ldof] =val 
                        
        elif space[0].doforder == 'vdims':
            for i in range(GD):
                val = np.einsum('i, ijm, ijn, j->jmn', ws, phi, gradphi[..., i], cellmeasure, optimize=True)
                K[:, i::GD, i::GD] += val 
        if out is None:
            return K


    def assembly_cell_matrix_fast(self, space, index=np.s_[:], cellmeasure=None):
        pass


    def assembly_cell_matrix_ref(self, space, index=np.s_[:], cellmeasure=None):
        pass
