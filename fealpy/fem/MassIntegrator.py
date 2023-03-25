import numpy as np


class MassIntegrator:
    """
    @note (c u, v)
    """    

    def __init__(self, c=None, q=3):
        self.coef = c
        self.q = q

    def assembly_cell_matrix(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        @note 没有参考单元的组装方式
        """
        q = self.q
        coef = self.coef
        mesh = space0.mesh

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        ps = mesh.bc_to_point(bcs, index=index)

        phi0 = space0.basis(bcs, index=index) # (NQ, NC, ldof, ...)
        phi1 = phi0

        if coef is None:
            M = np.einsum('q, qci, qcj, c->cij', ws, phi0, phi0, cellmeasure, optimize=True)
        else:
            if callable(coef):
                if hasattr(coef, 'coordtype'):
                    if coef.coordtype == 'barycentric':
                        coef = coef(bcs)
                    elif coef.coordtype == 'cartesian':
                        coef = coef(ps)
                else:
                    raise ValueError('''
                    You should add decorator "cartesian" or "barycentric" on
                    function `basis0`

                    from fealpy.decorator import cartesian, barycentric

                    @cartesian
                    def basis0(p):
                        ...

                    @barycentric
                    def basis0(p):
                        ...

                    ''')

            if np.isscalar(coef):
                M = np.einsum('q, qci, qcj, c->cij', ws, phi0, phi0, cellmeasure, optimize=True)
                M*=coef
            elif isinstance(coef, np.ndarray): 
                M = np.einsum('q, qc, qci, qcj, c->cij', ws, coef, phi0, phi0, cellmeasure, optimize=True)
            else:
                raise ValueError("coef is not correct!")

        return M

    def assembly_cell_matrix_fast(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        @brief 基于无数值积分的组装方式
        """
        mesh = space0.mesh 
        assert mesh.meshtype in ['tri', 'tet']

    def assembly_cell_matrix_ref(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        @note 基于参考单元的矩阵组装
        """
        pass
