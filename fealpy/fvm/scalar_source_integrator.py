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
        phi = space.basis(bcs, index=index) 
        return bcs, ws, cm, index, phi

    @variantmethod
    def assembly(self, space: _FS, indices=None) -> TensorLike:
        source = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, ws, cm, index, phi = self.fetch(space, indices)
        val = process_coef_func(source, bcs=bcs, mesh=mesh, etype='cell', index=index)
        D = phi.shape[-1]
        if D ==1:
            integrator = bm.einsum('j, qj,q -> q', ws, val, cm)
            return integrator
        elif D == 2:
            val1 = val[...,0]
            val2 = val[...,1]
            integrator1 = bm.einsum('j, qj,q -> q', ws, val1, cm)
            integrator2 = bm.einsum('j, qj,q -> q', ws, val2, cm)
            integrator = bm.stack((integrator1, integrator2), axis=-1)
            return integrator
        elif D == 3:
            val1 = val[...,0]
            val2 = val[...,1]
            val3 = val[...,1]
            integrator1 = bm.einsum('j, qj,q -> q', ws, val1, cm)
            integrator2 = bm.einsum('j, qj,q -> q', ws, val2, cm)
            integrator3 = bm.einsum('j, qj,q -> q', ws, val3, cm)
            integrator = bm.stack((integrator1, integrator2, integrator3), axis=-1)
            return integrator
        else:
            raise TypeError(f"source should be int, float or TensorLike, but got {type(source)}.")
        