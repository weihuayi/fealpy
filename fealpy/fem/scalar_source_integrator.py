from typing import Optional, Literal

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, SourceLike

from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import linear_integral
from ..decorator.variantmethod import variantmethod
from .integrator import LinearInt, SrcInt, CellInt, enable_cache


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
        q = self.q
        index = self.entity_selection(inidces)
        mesh = getattr(space, 'mesh', None)

        # if not isinstance(mesh, HomogeneousMesh):
        #     raise RuntimeError("The ScalarSourceIntegrator only support spaces on"
        #                        f"homogeneous meshes, but {type(mesh).__name__} is"
        #                        "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)

        return bcs, ws, phi, cm, index

    @variantmethod
    def assembly(self, space: _FS, indices=None) -> TensorLike:
        f = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space, indices)
        val = process_coef_func(f, bcs=bcs, mesh=mesh, etype='cell', index=index)
  
        return linear_integral(phi, ws, cm, val, batched=self.batched)

    @assembly.register('isopara')
    def assembly(self, space: _FS) -> TensorLike: 
        f = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)
        NQ = len(ws)
        NC = mesh.number_of_cells()

        rm = space.mesh.reference_cell_measure()
        J = space.mesh.jacobi_matrix(bcs)
        G = space.mesh.first_fundamental_form(J)
        d = bm.sqrt(bm.linalg.det(G))

        val = process_coef_func(f, bcs=bcs, mesh=mesh, etype='cell', index=index)
        if val is None:
            return bm.einsum('q, cql, cq -> cl ', ws*rm, phi, d) 

        if isinstance(val, (int, float)):
            return bm.einsum('q, cql, cq -> cl', ws*rm, phi, d)*val
        elif isinstance(val, TensorLike):
            if val.shape == (NC, ): # 分片常数
                return bm.einsum('q, c, cql, cq -> cl', ws*rm, val, phi, d)
            elif val.shape == (NC, NQ):
                return bm.einsum('q, cq, cql, cq -> cl', ws*rm, val, phi, d)
            else:
                raise ValueError(f"I can not deal with {f.shape}!")
