import numpy as np


class DiffusionIntegrator:
    """
    @note (c \grad u, \grad v)
    """    
    def __init__(self, c=None, q=3):
        self.coef = c
        self.q = q

    def assembly_cell_matrix(self, space, _, index=np.s_[:], cellmeasure=None):
        """
        @note 没有参考单元的组装方式
        """
        coef = self.coef
        q = self.q
        mesh = space.mesh
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()


        phi0 = space.grad_basis(bcs, index=index) # (NQ, NC, ldof, ...)
        phi1 = phi0

        if coef is None:
            D = np.einsum('q, qci..., qcj..., c->cij', ws, phi0, phi1, cellmeasure, optimize=True)
        else:
            if callable(coef):
                if hasattr(coef, 'coordtype'):
                    if coef.coordtype == 'barycentric':
                        coef = coef(bcs)
                    elif coef.coordtype == 'cartesian':
                        ps = mesh.bc_to_point(bcs)
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
                D = np.einsum('q, qci..., qcj..., c->cij', ws, phi0, phi1, cellmeasure, optimize=True)
                D*=coef
            elif coef.ndim == 2: #(NQ, NC)
                D = np.einsum('q, qc, qci..., qcj..., c->cij', ws, coef, phi0, phi1, cellmeasure, optimize=True)
            elif coef.ndim == 4:# (NQ, NC, GD, GD)
                phi0 = np.einsum('qcln, qcin->qcil', coef, phi0)
                D = np.einsum('q, qci..., qcj..., c->cij', ws, phi0, phi1,
                        cellmeasure, optimize=True)
            else:
                raise ValueError("coef 的维度超出了支持范围")

        return D

    def fast_assembly_cell_matrix(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        """
        mesh = space.mesh 
        assert mesh.meshtype in ['tri', 'tet']


    def assembly_cell_matrix_ref(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        @note 基于参考单元矩阵组装方式
        """
