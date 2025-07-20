from typing import Optional, Literal
from ..backend import backend_manager as bm
from ..typing import TensorLike, SourceLike
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..decorator.variantmethod import variantmethod
from ..fem.integrator import LinearInt, SrcInt, CellInt, enable_cache


class ScalarSourceIntegrator(LinearInt, SrcInt, CellInt):
    r"""The domain source integrator for function spaces based on homogeneous meshes."""
    def __init__(self, source: Optional[SourceLike]=None, q: int=None, *,
                 region: Optional[TensorLike] = None,
                 batched: bool=False,
                 method: Literal['isopara', None] = None) -> None:
        super().__init__()
        self.source = source
        self.q = q
        self.set_region(region)
        self.batched = batched
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        if indices is None:
            return space.cell_to_dof()
        return space.cell_to_dof(index=self.entity_selection(indices))

    @enable_cache
    def fetch(self, space: _FS, /, inidces=None):
        index = self.entity_selection(inidces)
        mesh = getattr(space, 'mesh', None)
        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        if isinstance(bcs, tuple): 
            bcs = bm.stack(bcs, axis=-1)
            ws = bm.stack(ws, axis=-1)  
        return bcs, ws, cm, index

    @variantmethod
    def assembly(self, space: _FS, indices=None) -> TensorLike:
        f = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, ws, cm, index = self.fetch(space, indices)
        val = process_coef_func(f, bcs=bcs, mesh=mesh, etype='cell', index=index)
        result = bm.einsum('j, qj,q -> q', ws, val, cm)
        return result 