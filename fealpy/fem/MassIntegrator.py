import numpy as np


class MassIntegrator:

    def __init__(self, c=None, q=3):
        self.coef = c
        self.q = q

    def assembly_cell_matrix(self, space0, _, index=np.s_[:], cellmeasure=None):
        """
        """
        q = self.q
        coef = self.coef
        mesh = space0.mesh

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        ps = mesh.bc_to_point(bcs, index=index)

        phi0 = b0(bcs, index=index) # (NQ, NC, ldof, ...)

        if coef is None:
            M = np.einsum('q, qci..., qcj..., c->cij', ws, phi0, phi0, cellmeasure, optimize=True)
        else:
            if callable(coef):
                if hasattr(c, 'coordtype'):
                    if coef.coordtype == 'barycentric':
                        c = coef(bcs)
                    elif coef.coordtype == 'cartesian':
                        c = coef(ps)
                else:
                    raise ValueError('''''')

            M = np.einsum('q, qc, qci..., qcj..., c->cij', ws, c, phi0, phi0, cellmeasure, optimize=True)

        return M

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
