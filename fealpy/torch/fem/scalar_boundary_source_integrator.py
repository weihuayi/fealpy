
from typing import Optional, Callable

from torch import Tensor

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import linear_integral, integral
from .integrator import FaceSourceIntegrator, enable_cache, CoefLike

# TODO: Support the threshold, process the index
class ScalarBoundarySourceIntegrator(FaceSourceIntegrator):
    r"""The boundary source integrator for function spaces based on homogeneous meshes."""
    def __init__(self, source: Optional[CoefLike]=None, q: int=3, *,
                 threshold: Optional[Callable[[Tensor], Tensor]]=None,
                 zero_integral: bool=False,
                 batched: bool=False) -> None:
        super().__init__()
        self.source = source
        self.q = q
        self.zero_integral = zero_integral
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> Tensor:
        index = space.mesh.boundary_face_index()
        return space.face_to_dof()[index]

    @enable_cache
    def fetch(self, space: _FS):
        q = self.q
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarBoundarySourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        index = mesh.boundary_face_index()
        fm = mesh.entity_measure('face', index=index)
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index, variable='x')

        return bcs, ws, phi, fm, index

    def assembly(self, space: _FS) -> Tensor:
        f = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, fm, index = self.fetch(space)
        val = process_coef_func(f, bcs=bcs, mesh=mesh, etype='face', index=index) # TODO: support normal derivative

        if self.zero_integral:
            if not isinstance(val, Tensor):
                raise RuntimeError("The zero_integral option is only supported when the source is a tensor.")
            val_int = integral(val, ws, fm/fm.sum())
            val = val - val_int.reshape((-1,) + (1,) * (val.ndim - 1))
            self._source_int = val_int

        self._source_val = val

        return linear_integral(phi, ws, fm, val, batched=self.batched)

    def get_source_val(self):
        """Get the latest source value
        (after the zero-integration if `zero_integral` is True).

        Returns:
            Tensor: Source value, shaped [Batch, Q, F_bd].
        """
        return self._source_val

    def get_source_int(self) -> Tensor:
        """Get the latest source integration on the boundary.
        This is available only when `zero_integral` is True.

        Returns:
            Tensor: Integration value of each sample, shaped [Batch,].
        """
        return self._source_int
