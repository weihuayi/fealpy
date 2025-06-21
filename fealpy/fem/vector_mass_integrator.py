
from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral, linear_integral, get_semilinear_coef
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache
)

class VectorMassIntegrator(LinearInt, OpInt, CellInt):
    """
    @note (c u, v)
    """    
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None) -> None:
        super().__init__()
        self.coef = coef
        self.q = q

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()

    @enable_cache
    def fetch(self, space: _FS):
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        return bcs, ws, phi, cm, index

    def assembly0(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        return bilinear_integral(phi, phi, ws, cm, val, batched=self.batched)

    def assembly(self, space, index=_S, cellmeasure=None, out=None):
        """
        @note 没有参考单元的组装方式
        """
        return self.assembly_cell_matrix_for_vector_basis_vspace(space, index=index, cellmeasure=cellmeasure, out=out)

    def assembly_cell_matrix_for_vector_basis_vspace(self, space, index=_S, cellmeasure=None, out=None):
        """
        @brief 空间基函数是向量型
        """
        q = space.p+3 if self.q is None else self.q

        coef = self.coef
        mesh = space.mesh
        GD = mesh.geo_dimension()
        self.device = mesh.device

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        NC = len(cellmeasure)
        ldof = space.number_of_local_dofs() 
        if out is None:
            D = bm.zeros((NC, ldof, ldof), device=self.device,dtype=space.ftype)
        else:
            D = out

        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        phi0 = space.basis(bcs) # (NQ, NC, ldof, GD)
        print(phi0.shape)

        import time

        start = time.time()
        if coef is None:
            D += bm.einsum('q, cqli, cqmi, c->clm', ws, phi0, phi0, cellmeasure)
        elif isinstance(coef, (int, float)):
            D += coef*bm.einsum('q, cqli, cqmi, c->clm', ws, phi0, phi0,
                                #cellmeasure, optimize=True)
                                cellmeasure)
        end = time.time()
        print(f"Time: {end-start}")

        if out is None:
            return D
        else:
            out += D


