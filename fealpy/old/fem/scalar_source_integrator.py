import numpy as np
from numpy.typing import NDArray

from typing import TypedDict, Callable, Tuple, Union


class ScalarSourceIntegrator():

    def __init__(self, f: Union[Callable, int, float, NDArray], q=None):
        """
        @brief

        @param[in] f 
        """
        self.f = f
        self.q = q
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

        q = p+3 if q is None else q

        mesh = space.mesh
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.dof.number_of_local_dofs() 
        if out is None:
            bb = np.zeros((NC, ldof), dtype=space.ftype)
        else:
            bb = out

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = space.basis(bcs, index=index) #TODO: 考虑非重心坐标的情形

        if callable(f):
            if hasattr(f, 'coordtype'):
                if f.coordtype == 'barycentric':
                    val = f(bcs, index=index)
                elif f.coordtype == 'cartesian':
                    ps = mesh.bc_to_point(bcs, index=index)
                    val = f(ps)
            else: # 默认是笛卡尔
                ps = mesh.bc_to_point(bcs, index=index)
                val = f(ps)
        else:
            val = f
        if isinstance(val, (int, float)):
            bb += val*np.einsum('q, qci, c->ci', ws, phi, cellmeasure, optimize=True)
        else:
            if val.shape == (NC, ): 
                bb += np.einsum('q, c, qci, c->ci', ws, val, phi, cellmeasure, optimize=True)
            else:
                if val.shape[-1] == 1:
                    val = val[..., 0]
                bb += np.einsum('q, qc, qci, c->ci', ws, val, phi, cellmeasure, optimize=True)
        if out is None:
            return bb 
        
