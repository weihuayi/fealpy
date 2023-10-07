import numpy as np


class DiffusionIntegrator:
    """
    @note (c \\grad u, \\grad v)
    """    
    def __init__(self, c=None, q=3):
        self.coef = c
        self.q = q

    def assembly_cell_matrix(self, space, index=np.s_[:], cellmeasure=None,
            out=None):
        """
        @note 没有参考单元的组装方式
        """
        coef = self.coef
        q = self.q
        if not isinstance(space, tuple): 
            space0 = space
        else:
            GD = len(space)
            space0 = space[0]
        
        mesh = space0.mesh
        GD = mesh.geo_dimension()
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)
        NC = len(cellmeasure)
        ldof = space0.number_of_local_dofs() 

        D = np.zeros((NC, ldof, ldof), dtype=space0.ftype)

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        phi0 = space0.grad_basis(bcs, index=index) # (NQ, NC, ldof, ...)
        phi1 = phi0

        if coef is None:
            D += np.einsum('q, qci..., qcj..., c->cij', ws, phi0, phi1, cellmeasure, optimize=True)
        else:
            if callable(coef):
                if hasattr(coef, 'coordtype'):
                    if coef.coordtype == 'cartesian':
                        ps = mesh.bc_to_point(bcs, index=index)
                        coef = coef(ps)
                    elif coef.coordtype == 'barycentric':
                        coef = coef(bcs, index=index)
                else:
                    ps = mesh.bc_to_point(bcs, index=index)
                    coef = coef(ps)
            if np.isscalar(coef):
                D += coef*np.einsum('q, qci..., qcj..., c->cij', ws, phi0, phi1, cellmeasure, optimize=True)
            elif isinstance(coef, np.ndarray): 
                if coef.shape == (NC, ): 
                    D += np.einsum('q, c, qcim, qcjm, c->cij', ws, coef, phi0, phi1, cellmeasure, optimize=True)
                elif coef.shape == (NQ, NC):
                    D += np.einsum('q, qc, qcim, qcjm, c->cij', ws, coef, phi0, phi1, cellmeasure, optimize=True)
                else:
                    n = len(coef.shape)
                    shape = (4-n)*(1, ) + coef.shape
                    D += np.einsum('q, qcmn, qcin, qcjm, c->cij', ws, coef.reshape(shape), phi0, phi1, cellmeasure, optimize=True)
            else:
                raise ValueError("coef不支持该类型")

        if not isinstance(space, tuple): 
            if out is None:
                return D
            else:
                assert out.shape == (NC, ldof, ldof)
                out += D
        else:
            if out is None:
                VD = n.zeros((NC, GD*ldof, GD*ldof), dtype=space.ftype)
            else:
                assert out.shape == (NC, GD*ldof, GD*ldof)
                VD = out
            
            if space0.doforder == 'sdofs':
                for i in range(GD):
                    VD[:, i*ldof:(i+1)*ldof, i*ldof:(i+1)*ldof] += D
            elif space0.doforder == 'vdims':
                for i in range(GD):
                    VD[:, i::GD, i::GD] += D 
            
            if out is None:
                return VM


    def assembly_cell_matrix_fast(self, space, index=np.s_[:], cellmeasure=None):
        """
        """
        mesh = space0.mesh 
        assert mesh.meshtype in ['tri', 'tet']


    def assembly_cell_matrix_ref(self, space, index=np.s_[:], cellmeasure=None):
        """
        @note 基于参考单元矩阵组装方式
        """
        pass
