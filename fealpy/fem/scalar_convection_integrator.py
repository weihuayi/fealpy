import numpy as np

class ScalarConvectionIntegrator:
    """
    @note ( c \\cdot \\nabla u, v)
    """    

    def __init__(self, c=None, q=3):
        self.coef = c
        self.q = q

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None,
            out=None):

        q = self.q
        coef = self.coef
        mesh = space.mesh
        
        GD = mesh.geo_dimension()
        if cellmeasure is None:
            if mesh.meshtype == 'UniformMesh2d':
                 NC = mesh.number_of_cells()
                 cellmeasure = np.broadcast_to(mesh.entity_measure('cell', index=index), (NC,))
            else:
                cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs() 
        
        if out is None:
            C = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        else:
            C = out
        
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)
        
        gphi = space.grad_basis(bcs, index=index) 
        phi = space.basis(bcs, index=index) 
        
        if callable(coef):
            if hasattr(coef, 'coordtype'):
                if coef.coordtype == 'barycentric':
                    coef = coef(bcs, index)
                elif coef.coordtype == 'cartesian':
                    ps = mesh.bc_to_point(bcs, index=index)
                    coef = coef(ps)
            else:
                ps = mesh.bc_to_point(bcs, index=index)
                coef = coef(ps)
        if coef.shape == (GD, ):
            C += np.einsum('q, qck, n, qcmn, c->ckm', ws, phi, coef, gphi, cellmeasure) 
        elif coef.shape == (NC, GD):
            C += np.einsum('q, qck, cn, qcmn, c->ckm', ws, phi, coef, gphi, cellmeasure) 
        elif coef.shape == (NQ, NC, GD): 
            C += np.einsum('q, qck, qcn, qcmn, c->ckm', ws, phi, coef, gphi, cellmeasure) 
        elif coef.shape == (NQ, GD, NC): 
            C += np.einsum('q, qck, qnc, qcmn, c->ckm', ws, phi, coef, gphi, cellmeasure) 
        else:
            raise ValueError("coef 的维度超出了支持范围")

        if out is None:
            return C

    def assembly_cell_matrix_fast(self, space, index=np.s_[:], cellmeasure=None):
        """
        """
        mesh = space.mesh 
        assert mesh.meshtype in ['tri', 'tet']

    def assembly_cell_matrix_ref(self, space, index=np.s_[:], cellmeasure=None):
        """
        @note 基于参考单元的矩阵组装
        """
        pass
