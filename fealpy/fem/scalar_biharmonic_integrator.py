
from typing import Optional, Literal

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral, linear_integral, get_semilinear_coef
from ..decorator.variantmethod import variantmethod
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache
)

class ScalarBiharmonicIntegrator(LinearInt, OpInt, CellInt):
    r"""The biharmonic integrator for function spaces based on homogeneous meshes."""
    def __init__(self, coef: Optional[CoefLike] = None, q: Optional[int] = None, *,
                 index: Index = _S,
                 batched: bool = False,
                 method: Literal['fast', 'nonlinear', 'isopara', None] = None) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarDiffusionIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        return bcs, ws, cm

    @enable_cache
    def fetch_hphix(self, space: _FS):
        bcs = self.fetch(space)[0]
        return space.hess_basis(bcs, index=self.index, variable='x')

    @enable_cache
    def fetch_hphiu(self, space: _FS):
        bcs = self.fetch(space)[0]
        return space.hess_basis(bcs, index=self.index, variable='u')

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, cm = self.fetch(space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=self.index)
        hphi = self.fetch_hphix(space)
        
        return bilinear_integral(hphi, hphi, ws, cm, coef, batched=self.batched)
    
    @assembly.register('fast')
    def assembly(self, space: _FS) -> TensorLike:
        """
        限制：常系数、单纯形网格
        TODO: 加入 assert
        """
        pass
