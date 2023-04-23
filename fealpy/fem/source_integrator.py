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

        @todo 考虑向量和张量空间的情形
        
        @note 该函数有如下的情形需要考虑： 
            * f 是标量 
            * f 是标量函数 (NQ, NC)，基是标量函数 (NQ, NC, ldof)
            * f 是向量函数 (NQ, NC, GD)， 基是向量函数 (NQ, NC, ldof, GD)
        """
        f = self.f
        q = self.q


        mesh = space.mesh
        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs() 
        if out is None:
            bb = np.zeros((NC, gdof), dtype=space.ftype)
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
            bb += val*np.einsum('q, qc, qci, c->ci', ws, phi, cellmeasure, optimize=True)
        elif isinstance(val, np.ndarray): 
            if val.shape[-1] == 1:
                val = val[..., 0]
            bb += np.einsum('q, qc, qci, c->ci', ws, val, phi, cellmeasure, optimize=True)
        else:
            raise ValueError("We need to consider more cases!")

        if out is None:
            return bb 
        
