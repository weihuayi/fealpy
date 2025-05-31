from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, SourceLike

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from .integrator import LinearInt, SrcInt, CellInt, enable_cache


class GradSourceIntegrator(LinearInt, SrcInt, CellInt):
    r"""<f, grad v>"""
    def __init__(self, source: Optional[SourceLike]=None, q: int=None, *,
                 region: Optional[TensorLike] = None,
                 batched: bool=False) -> None:
        super().__init__()
        self.source = source
        self.q = q
        self.set_region(region)
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        if indices is None:
            return space.cell_to_dof()
        return space.cell_to_dof(index=self.entity_selection(indices))
    
    @enable_cache
    def fetch(self, space: _FS, /, inidces=None):
        q = self.q
        index = self.entity_selection(inidces)
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarSourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index=index)
        return bcs, ws, gphi, cm, index

    def assembly(self, space: _FS, indices=None) -> TensorLike:
        f = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, ws, gphi, cm, index = self.fetch(space, indices)
        val = process_coef_func(f, bcs=bcs, mesh=mesh, etype='cell', index=index) 
        #result = linear_integral(gphi, ws, cm, val, batched=self.batched)
        result = bm.einsum('q,c,cq...,cql...->cl', ws, cm, val, gphi)
        return result
