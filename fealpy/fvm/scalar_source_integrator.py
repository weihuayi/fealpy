from typing import Optional, Literal

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, SourceLike
from fealpy.utils import process_coef_func
from fealpy.decorator import variantmethod

from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import LinearInt, SrcInt, CellInt, enable_cache


class ScalarSourceIntegrator(LinearInt, SrcInt, CellInt):
    r"""The domain source integrator for function spaces based on homogeneous meshes."""
    def __init__(self, source: Optional[SourceLike]=None, q: int=None, *,
                 region: Optional[TensorLike] = None,
                 batched: bool=False,
                 method: Literal['isopara', None] = None) -> None:
        super().__init__()
        self.source = source
        self.q = 2 if q is None else q
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
        qf = mesh.quadrature_formula(self.q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights() 
        return bcs, ws, cm, index

    @variantmethod
    def assembly(self, space: _FS, indices=None) -> TensorLike:
        source = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, ws, cm, index = self.fetch(space, indices)
        val = process_coef_func(
            source, bcs=bcs, mesh=mesh, etype='cell', index=index
        )
        # val: (Q, nq) or (Q, nq, D)
        if val.ndim == 2:
            # 标量源项 -> (Q,)
            return bm.einsum('j, qj, q -> q', ws, val, cm)
        elif val.ndim == 3:
            # 向量源项 -> (Q, D)
            return bm.einsum('j, qjd, q -> qd', ws, val, cm)
        else:
            raise ValueError(f"Unsupported source shape: {val.shape}")
            