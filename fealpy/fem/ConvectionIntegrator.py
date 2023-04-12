import numpy as np


class ConvectionIntegrator:
    """
    @note (a * c \\cdot \\nabla u, v)
    """    

    def __init__(self, c=None, a=1 ,q=3):
        self.coef = c
        self.a = a
        self.q = q

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None,
            out=None):
        """
        @note 没有参考单元的组装方式
        """
        q = self.q
        coef = self.coef
        mesh = space.mesh
        
        GD = mesh.geo_dimension()
        if cellmeasure is None:
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
            
            if coef.shape == (NQ, NC, GD):  
                #print(np.sum(np.abs(coef)))
                C += self.a*np.einsum('q, qcn, qck, qcmn, c->ckm',ws, coef, phi, gphi, cellmeasure) 
            elif coef.shape == (NC, GD):
                C += self.a*np.einsum('q, cn, qck, qcmn, c->ckm',ws, coef, phi, gphi, cellmeasure) 
            else:
                raise ValueError("coef 的维度超出了支持范围")

        return C

    def fast_assembly_cell_matrix(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        """
        mesh = space.mesh 
        assert mesh.meshtype in ['tri', 'tet']

    def assembly_cell_matrix_ref(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        @note 基于参考单元的矩阵组装
        """
        pass
