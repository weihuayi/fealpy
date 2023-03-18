import numpy as np


class MassIntegrator:

    def assembly_cell_matrix(self, mesh, b0, b1=None, c=None, q=3, cellmeasure=None):
        """
        """
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell')

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        ps = mesh.bc_to_point(bcs)

        phi0 = b0(bcs) # (NQ, NC, ldof, ...)

        if b1 is not None:
            phi1 = b1(bcs) # (NQ, NC, ldof, ...)
        else:
            phi1 = phi0

        if c is None:
            M = np.einsum('q, qci..., qcj..., c->cij', ws, phi0, phi1, self.cellmeasure, optimize=True)
        else:
            if callable(c):
                if hasattr(c, 'coordtype'):
                    if c.coordtype == 'barycentric':
                        c = c(bcs)
                    elif c.coordtype == 'cartesian':
                        c = c(ps)
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

            M = np.einsum('q, qc, qci..., qcj..., c->cij', ws, c, phi0, phi1, self.cellmeasure, optimize=True)

        return M
