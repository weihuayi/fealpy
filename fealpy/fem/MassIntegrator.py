import numpy as np


class MassIntegrator:

    def __init__(self, c=None, q=3):
        self.coef = c
        self.q = q

    def assembly_cell_matrix(self, mesh, b0, index=np.s_[:], cellmeasure=None):
        """
        """
        q = self.q
        c = self.coef

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        ps = mesh.bc_to_point(bcs, index=index)

        phi0 = b0(bcs, index=index) # (NQ, NC, ldof, ...)

        if c is None:
            M = np.einsum('q, qci..., qcj..., c->cij', ws, phi0, phi0, cellmeasure, optimize=True)
        else:
            if callable(c):
                if hasattr(c, 'coordtype'):
                    if c.coordtype == 'barycentric':
                        c = c(bcs)
                    elif c.coordtype == 'cartesian':
                        c = c(ps)
                else:
                    raise ValueError('''''')

            M = np.einsum('q, qc, qci..., qcj..., c->cij', ws, c, phi0, phi0, cellmeasure, optimize=True)

        return M

    def fast_assembly_cell_matrix(self, mesh, p):
        """
        """
        assert mesh.meshtype in ['tri', 'tet']
