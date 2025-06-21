import numpy as np
from ..quadrature import GaussLegendreQuadrature
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

class ScaledMonomialSpaceDiffusionIntegrator2d:
    def __init__(self, q=3):
        self.q = q

    def assembly_cell_matrix(self, space: ScaledMonomialSpace2d, M):
        p = space.p
        mesh = space.mesh
        NC = mesh.number_of_cells()
        ldof0 = space.number_of_local_dofs(p=p-1)

        ldof = space.number_of_local_dofs()
        assert M.shape == (NC, ldof, ldof)

        cm = mesh.entity_measure('cell')

        dindex = space.diff_index_1()
        ix, cx = dindex['x']
        iy, cy = dindex['y']

        G = np.zeros((NC, ldof, ldof), dtype=space.ftype)

        idx = np.arange(ldof0)

        C = cx[:, None]*cx
        G[:, ix[:, None], ix] += M[:, idx[:, None], idx]*C

        C = cy[:, None]*cy
        G[:, iy[:, None], iy] += M[:, idx[:, None], idx]*C

        G /= cm[:, None, None]

        return G

class ScaledMonomialSpaceDiffusionIntegrator3d:
    def __init__(self, q=3):
        self.q = q

    def assembly_cell_matrix(self, space: ScaledMonomialSpace2d, M=None):
        pass
