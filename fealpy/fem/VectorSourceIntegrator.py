import numpy as np
from numpy.typing import NDArray

from typing import TypedDict, Callable, Tuple, Union


class VectorSourceIntegrator():

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

        if isinstance(space, tuple):
            mesh = space[0].mesh # 向量空间是由标量空间组合而成
        else:
            mesh = space.mesh # 向量空间的基函数是向量函数

        GD = mesh.geo_dimension()
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
        if isinstance(space, tuple):
            phi = space[0].basis(bcs, index=index)
        else:
            phi = space.basis(bcs, index=index)

        if callable(f):
            if hasattr(f, 'coordtype'):
                if f.coordtype == 'cartesian':
                    ps = mesh.bc_to_point(bcs, index=index)
                    val = f(ps)
                elif f.coordtype == 'barycentric':
                    val = f(bcs, index=index)
            else: # 默认是笛卡尔
                ps = mesh.bc_to_point(bcs, index=index)
                val = f(ps)
        else:
            val = f

        if not isinstance(space, tuple):
            bb += np.einsum('q, qcd, qcid, c->ci', ws, val, phi, cellmeasure, optimize=True)
        else:
            if space[0].doforder == 'sdofs':
                bb += np.einsum('q, qcd, qci, c->cdi', ws, val, phi, cellmeasure, optimize=True)
            elif space[0].doforder == 'vdims':
                bb += np.einsum('q, qcd, qci, c->cid', ws, val, phi, cellmeasure, optimize=True)
            else:
                raise ValueError(f"we don't support the space[0].doforder type: {space[0].doforder}")

        if out is None:
            return bb 
        
