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
        mesh = space.mesh
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        if index != np.s_[:]:
            NC = len(index)


        phi0 = space0.grad_basis(bcs, index=index) # (NQ, NC, ldof, ...)
        phi1 = phi0

        if coef is None:
            D = np.einsum('q, qci..., qcj..., c->cij', ws, phi0, phi1, cellmeasure, optimize=True)
        else:
            if callable(coef):
                if hasattr(coef, 'coordtype'):
                    if coef.coordtype == 'barycentric':
                        coef = coef(bcs, index=index)
                    elif coef.coordtype == 'cartesian':
                        ps = mesh.bc_to_point(bcs, index=index)
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
            elif isinstance(coef, np.ndarray): 
                if coef.shape == (NC, ): 
                    D = np.einsum('q, c, qcim, qcjm, c->cij', ws, coef, phi0, phi1, cellmeasure, optimize=True)
                elif coef.shape == (NQ, NC):
                    D = np.einsum('q, qc, qcim, qcjm, c->cij', ws, coef, phi0, phi1, cellmeasure, optimize=True)
                else:
                    n = len(coef.shape)
                    shape = (4-n)*(1, ) + coef.shape
                    D = np.einsum('q, qcmn, qcin, qcjm, c->cij', ws, coef.reshape(shape), phi0, phi1, cellmeasure, optimize=True)
            else:
                raise ValueError("coef不支持该类型")

        return D

    def assembly_cell_matrix_fast(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        """
        mesh = space0.mesh 
        assert mesh.meshtype in ['tri', 'tet']


    def assembly_cell_matrix_ref(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        @note 基于参考单元矩阵组装方式
        """
