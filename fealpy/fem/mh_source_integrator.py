import numpy as np
from numpy.typing import NDArray
from typing import TypedDict, Callable, Tuple,Union

class ScalarSourceIntegrator():

    def __init__(self, f: Union[Callable, int, float, NDArray], C, q=None):
        """
        @brief

        @param[in] f 
        """
        self.f = f
        self.q = q
        self.C = C
        self.vector = None

    def assembly_cell_vector(self, 
            space, 
            index=np.s_[:], 
            cellmeasure=None,
            out=None):
        """
        @brief 组装单元向量

        @param[in] space 一个标量的函数空间

        """
        f = self.f
        p = space.p
        q = self.q
        C = self.C

        q = p+3 if q is None else q

        mesh = space.mesh
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs() 
        if out is None:
            bb = np.zeros((NC, ldof), dtype=space.ftype)
        else:
            bb = out



        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = space.basis(bcs, index=index) #TODO: 考虑非重心坐标的情形

        ps = mesh.bc_to_point(bcs, index=index)
        val = f(ps)
        bb += np.einsum('q, qc, qci, c->ci', ws, val, phi, cellmeasure, optimize=True)
        bb += np.einsum('q, qc, ci, c->ci', ws, val, C, cellmeasure, optimize=True)
        return bb 

