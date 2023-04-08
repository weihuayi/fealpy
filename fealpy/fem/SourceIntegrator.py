import numpy as np
from numpy.typing import NDArray

from typing import TypedDict, Callable, Tuple, Union


class SourceIntegrator():

    def __init__(self, 
            f: Union[Callable, int, float, NDArray], 
            q: int=3):
        """
        @brief

        @param[in] f 
        """
        self.f = f
        self.q = q
        self.vector = None

    def assembly_cell_vector(self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 组装单元向量

        @param[in] space 
        """
        f = self.f
        q = self.q


        mesh = space.mesh

        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        phi = space.basis(bcs) #@TODO: 考虑非重心坐标的情形
        if callable(f):
            if f.coordtype == 'cartesian':
                ps = mesh.bc_to_point(bcs)
                val = f(ps)
            elif f.coordtype == 'barycentric':
                val = f(bcs)

        if isinstance(f, np.ndarray):
            assert len(f) == NC
            val = f
        else:
            val = np.broadcast_to(f, shape=(NC, )) 

        bb = np.einsum('i, ijm, ijkm, j->jk', ws, val, phi, self.cellmeasure)

        return b
        
