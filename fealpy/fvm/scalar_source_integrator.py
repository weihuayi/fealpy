"""Cell source-term integrator for finite-volume right-hand sides."""

from typing import Optional, Literal

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, SourceLike
from fealpy.decorator import variantmethod

from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import LinearInt, SrcInt, CellInt, enable_cache

from .fvm_geometry import FVMGeometry


class ScalarSourceIntegrator(LinearInt, SrcInt, CellInt):
    r"""Integrate scalar or vector source terms over control volumes.

    The returned value is the cell-integrated right-hand side contribution,
    not a pointwise source value.  For scalar source data the result has shape
    ``(NC,)``; for vector source data the result has shape ``(NC, D)``.  The
    solver layer decides how vector components are flattened into a global
    algebraic vector.
    """

    def __init__(self, source: Optional[SourceLike]=None, q: int=None, *,
                 region: Optional[TensorLike] = None,
                 batched: bool=False,
                 method: Literal['isopara', None] = None,
                 geometry: Optional[FVMGeometry] = None) -> None:
        super().__init__()
        self.source = source
        self.q = 2 if q is None else q
        self.set_region(region)
        self.batched = batched
        self.geometry = geometry
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        if indices is None:
            return space.cell_to_dof()
        return space.cell_to_dof(
            index=self.entity_selection(indices, mesh=space.mesh)
        )

    @variantmethod
    def assembly(self, space: _FS, indices=None) -> TensorLike:
        source = self.source
        mesh = getattr(space, 'mesh', None)
        geometry = self.geometry or FVMGeometry(mesh)
        if geometry.mesh is not mesh:
            raise ValueError("geometry and function space must use the same mesh.")
        index = self.entity_selection(indices, mesh=mesh)

        if callable(source):
            def integrand(points, _cell_slice):
                values = bm.array(
                    source(points),
                    device=bm.get_device(points),
                )
                point_shape = tuple(points.shape[:-1])
                if values.ndim == 0:
                    return bm.ones(
                        point_shape,
                        dtype=points.dtype,
                        device=bm.get_device(points),
                    ) * values
                if tuple(values.shape[:len(point_shape)]) == point_shape:
                    return values
                if values.ndim == 1:
                    return bm.ones(
                        point_shape + (values.shape[0],),
                        dtype=points.dtype,
                        device=bm.get_device(points),
                    ) * values
                raise ValueError(
                    "callable source must return scalar or vector values at "
                    f"cell quadrature points; got shape {values.shape}."
                )

            values = geometry.cell_integral(integrand, q=self.q)
        else:
            if source is None:
                source = 0.0
            source = bm.array(
                source,
                dtype=geometry.cell_center.dtype,
                device=bm.get_device(geometry.cell_center),
            )
            if source.ndim == 0:
                values = geometry.cell_measure * source
            else:
                if source.shape[0] != geometry.NC:
                    raise ValueError(
                        "array source must have one value per global cell; "
                        f"got {source.shape[0]}, expected {geometry.NC}."
                    )
                measure_shape = (geometry.NC,) + (1,) * (source.ndim - 1)
                values = bm.reshape(geometry.cell_measure, measure_shape) * source

        return values[index]
