import numpy as np
from numpy.typing import NDArray

from typing import TypedDict, Callable, Tuple, Union, Optional


class VectorSourceIntegrator():

    def __init__(self, 
            f: Union[Callable, int, float, NDArray], 
            q: Optional[int]=None):
        """
        @brief

        @param[in] f 
        """
        self.f = f
        self.q = q

    def assembly_cell_vector(self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 组装单元向量

        @param[in] space 

        @note f 是一个向量函数，返回形状可以为
            * (GD, ) 常向量情形
            * (NC, GD) 分片常向量情形
            * (NQ, NC, GD) 变向量情形
        """

        if isinstance(space, tuple) and not isinstance(space[0], tuple):
            return self.assembly_cell_vector_for_vspace_with_scalar_basis(space, 
                    index=index, cellmeasure=cellmeasure, out=out)
        else:
            return self.assembly_cell_vector_for_vspace_with_vector_basis(space, 
                    index=index, cellmeasure=cellmeasure, out=out)
        

    def assembly_cell_vector_for_vspace_with_scalar_basis(self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 由标量空间张成的向量空间 

        @param[in] space 
        """
        # 假设向量空间是由标量空间组合而成
        assert isinstance(space, tuple) and not isinstance(space[0], tuple) 
        
        f = self.f
        mesh = space[0].mesh # 获取网格对像
        GD = mesh.geo_dimension()

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space[0].number_of_local_dofs() 
        if out is None:
            if space[0].doforder == 'sdofs': # 标量基函数自由度排序优先
                bb = np.zeros((NC, GD, ldof), dtype=space[0].ftype)
            elif space[0].doforder == 'vdims': # 向量分量自由度排序优先
                bb = np.zeros((NC, ldof, GD), dtype=space[0].ftype)
        else:
            bb = out

        q = self.q if self.q is not None else space[0].p + 1 
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        phi = space[0].basis(bcs, index=index)

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
        if isinstance(val, (int, float)):
            if space[0].doforder == 'sdofs':
                bb += val*np.einsum('q, qci, c->ci', ws, phi, cellmeasure, optimize=True)[:, None, :]
            elif space[0].doforder == 'vdims':
                bb += val*np.einsum('q, qci, c->ci', ws, val, phi, cellmeasure, optimize=True)[:, :, None]
        elif isinstance(val, np.ndarray):
            if val.shape == (GD, ): # GD << NC
                if space[0].doforder == 'sdofs':
                    bb += np.einsum('q, d, qci, c->cdi', ws, val, phi, cellmeasure, optimize=True)
                elif space[0].doforder == 'vdims':
                    bb += np.einsum('q, d, qci, c->cid', ws, val, phi, cellmeasure, optimize=True)
            elif val.shape == (NC, GD): 
                if space[0].doforder == 'sdofs':
                    bb += np.einsum('q, cd, qci, c->cdi', ws, val, phi, cellmeasure, optimize=True)
                elif space[0].doforder == 'vdims':
                    bb += np.einsum('q, cd, qci, c->cid', ws, val, phi, cellmeasure, optimize=True)
            elif val.shape == (NQ, NC, GD):
                if space[0].doforder == 'sdofs':
                    bb += np.einsum('q, qcd, qci, c->cdi', ws, val, phi, cellmeasure, optimize=True)
                elif space[0].doforder == 'vdims':
                    bb += np.einsum('q, qcd, qci, c->cid', ws, val, phi, cellmeasure, optimize=True)
            elif val.shape == (NQ, GD, NC):
                if space[0].doforder == 'sdofs':
                    bb += np.einsum('q, qdc, qci, c->cdi', ws, val, phi, cellmeasure, optimize=True)
                elif space[0].doforder == 'vdims':
                    bb += np.einsum('q, qdc, qci, c->cid', ws, val, phi, cellmeasure, optimize=True)
            else:
                raise ValueError("coef 的维度超出了支持范围")
        if out is None:
            return bb 

    def assembly_cell_vector_for_vspace_with_vector_basis(
            self, space, index=np.s_[:], cellmeasure=None, out=None):
        """
        @brief 组装单元向量

        @param[in] space 
        """

        space = self.space
        f = self.f

        mesh = space.mesh # 获取网格对像
        GD = mesh.geo_dimension()

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs() 

        if out is None:
            bb = np.zeros((NC, ldof), dtype=space.ftype)
        else:
            bb = out

        q = self.q if self.q is not None else space.p + 3 
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

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

        if isinstance(val, np.ndarray):
            if val.shape == (GD, ): # GD << NC
                bb += np.einsum('q, d, qcid, c->ci', ws, val, phi, cellmeasure, optimize=True)
            elif val.shape == (NC, GD): 
                bb += np.einsum('q, cd, qcid, c->ci', ws, val, phi, cellmeasure, optimize=True)
            elif val.shape == (NQ, NC, GD):
                bb += np.einsum('q, qcd, qcid, c->ci', ws, val, phi, cellmeasure, optimize=True)

        if out is None:
            return bb 
