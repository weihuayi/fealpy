
from typing import Optional, Callable

from torch import Tensor

from ..mesh import HomoMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import linear_integral, integral
from .integrator import FaceSourceIntegrator, _S, Index, CoefLike

# TODO: Support the threshold, process the index
class ScalarBoundarySourceIntegrator(FaceSourceIntegrator):
    r"""The boundary source integrator for function spaces based on homogeneous meshes."""
    def __init__(self, source: Optional[CoefLike]=None, q: int=3, *,
                 threshold: Optional[Callable[[Tensor], Tensor]]=None,
                 zero_integral: bool=False,
                 batched: bool=False) -> None:
        super().__init__()
        self.f = source
        self.q = q
        self.zero_integral = zero_integral
        self.batched = batched

    def to_global_dof(self, space: _FS) -> Tensor:
        index = space.mesh.ds.boundary_face_index()
        return space.face_to_dof()[index]

    def assembly(self, space: _FS) -> Tensor:
        f = self.f
        q = self.q
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomoMesh):
            raise RuntimeError("The ScalarBoundarySourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        index = mesh.ds.boundary_face_index()
        fm = mesh.entity_measure('face', index=index)
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index, variable='x')
        val = process_coef_func(f, bcs=bcs, mesh=mesh, etype='face', index=index) # TODO: support normal derivative

        if self.zero_integral:
            if not isinstance(val, Tensor):
                raise RuntimeError("The zero_integral option is only supported when the source is a tensor.")
            val_int = integral(val, ws, fm/fm.sum())
            val = val - val_int.reshape((-1,) + (1,) * (val.ndim - 1))

        return linear_integral(phi, ws, fm, val, batched=self.batched)
