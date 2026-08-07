"""Explicit finite-volume source for the deviatoric viscous-stress correction."""

from typing import Literal, Optional

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.fem.integrator import FaceInt, LinearInt, OpInt, enable_cache
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.typing import CoefLike, Index, TensorLike, _S

from .fvm_geometry import FVMGeometry


class DeviatoricStressSourceIntegrator(LinearInt, OpInt, FaceInt):
    r"""Assemble the explicit deviatoric viscous-stress correction RHS.

    For each selected face ``f``, ``grad_f[f, i, j]`` stores
    ``d u_i / d x_j`` and ``FVMGeometry.S_f`` stores the owner-oriented face
    area vector.  The vector face flux assembled here is

        F_f = mu_f (grad(U)_f^T - 2/3 div(U)_f I) S_f.

    The cell source is obtained by finite-volume face summation: ``F_f`` is
    added to the owner cell and subtracted from the neighbour cell.  This
    integrator only contributes the explicit deviatoric stress correction; the
    implicit orthogonal Laplacian part is assembled by the diffusion
    integrator.

    Future extensions should stay narrow: add dedicated 3D verification before
    using this route in three-dimensional flow solvers, and keep cell-wise or
    functional viscosity handling outside this integrator by interpolating it
    to face-wise ``coef`` before assembly.
    """

    def __init__(
        self,
        grad_f: TensorLike,
        coef: Optional[CoefLike] = None,
        *,
        index: Index = _S,
        region: Optional[Index] = None,
        geometry: Optional[FVMGeometry] = None,
        batched: bool = False,
        method: Literal[None] = None,
    ) -> None:
        super().__init__()
        if region is not None and not (isinstance(index, slice) and index == _S):
            raise ValueError("Use either 'region' or 'index', not both.")
        self.grad_f = grad_f
        self.coef = coef
        self.index = index
        self.geometry = geometry
        self.batched = batched
        self.set_region(region)
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        # The integration region is face-based, but the scattered source lives
        # on cell unknowns.
        return space.cell_to_dof()

    @enable_cache
    def fetch(self, space: _FS, /, indices=None):
        mesh = space.mesh
        full_geometry = self.geometry if self.geometry is not None else FVMGeometry(mesh)
        total_faces = full_geometry.NF

        region = self.get_region()
        if region is not None:
            if isinstance(region, slice) and indices is not None:
                index = bm.arange(
                    total_faces,
                    device=bm.get_device(full_geometry.owner),
                )[region][indices]
            else:
                index = self.entity_selection(indices, mesh=mesh)
        elif indices is None:
            index = self.index
        elif isinstance(self.index, slice) and self.index == _S:
            index = indices
        elif isinstance(self.index, slice):
            index = bm.arange(
                total_faces,
                device=bm.get_device(full_geometry.owner),
            )[self.index][indices]
        elif bm.is_tensor(self.index):
            if self.index.dtype == bm.bool:
                index = bm.nonzero(self.index)[0][indices]
            else:
                index = self.index[indices]
        else:
            raise TypeError(
                f"index of type '{self.index.__class__.__name__}' is not supported "
                "when local indices are given."
            )

        if (
            self.geometry is not None
            and region is None
            and isinstance(index, slice)
            and index == _S
        ):
            return full_geometry, index, total_faces

        return FVMGeometry(mesh, index=index), index, total_faces

    @variantmethod
    def assembly(self, space: _FS, /, indices=None) -> TensorLike:
        geometry, index, total_faces = self.fetch(space, indices)

        grad_f = bm.array(
            self.grad_f,
            dtype=geometry.cell_center.dtype,
            device=bm.get_device(geometry.cell_center),
        )
        if grad_f.shape[0] == total_faces:
            grad_f = grad_f[index]

        coef = self.coef
        if coef is not None and not isinstance(coef, (int, float)):
            coef = bm.array(
                coef,
                dtype=grad_f.dtype,
                device=bm.get_device(grad_f),
            )
            if coef.shape != () and coef.shape[0] == total_faces:
                coef = coef[index]

        face_flux = deviatoric_stress_face_flux(grad_f, geometry.S_f, coef)
        return geometry.scatter_face_flux_to_cells(face_flux)


def deviatoric_stress_face_flux(
    grad_f: TensorLike,
    face_normal: TensorLike,
    coef: Optional[CoefLike] = None,
) -> TensorLike:
    r"""Return owner-oriented face flux for ``dev2(T(grad(U)))``.

    ``grad_f[f, i, j]`` stores ``d u_i / d x_j`` and ``face_normal`` stores the
    owner-oriented face vector ``S_f``.  The face flux is

        mu_f [grad(U)_f^T - 2/3 div(U)_f I] S_f,
    """
    grad_f = bm.array(
        grad_f,
        device=bm.get_device(grad_f),
    )
    face_normal = bm.array(
        face_normal,
        dtype=grad_f.dtype,
        device=bm.get_device(grad_f),
    )

    if grad_f.ndim != 3 or grad_f.shape[1] != grad_f.shape[2]:
        raise ValueError(f"grad_f must have shape (NF, GD, GD), got {grad_f.shape}.")
    if face_normal.ndim != 2 or grad_f.shape[:2] != face_normal.shape:
        raise ValueError(
            "face_normal must have shape (NF, GD) matching grad_f, got "
            f"{face_normal.shape}."
        )

    if coef is None:
        coef_f = bm.ones_like(face_normal[:, 0], dtype=grad_f.dtype)
    elif isinstance(coef, (int, float)):
        coef_f = bm.full_like(face_normal[:, 0], fill_value=coef, dtype=grad_f.dtype)
    else:
        coef_f = bm.array(
            coef,
            dtype=grad_f.dtype,
            device=bm.get_device(grad_f),
        )
        if coef_f.shape == ():
            coef_f = bm.full_like(face_normal[:, 0], fill_value=float(coef_f), dtype=grad_f.dtype)
        elif coef_f.shape[0] != face_normal.shape[0]:
            raise ValueError(
                f"coef has incompatible first dimension {coef_f.shape[0]}; expected "
                f"{face_normal.shape[0]} faces."
            )

    div_u = bm.einsum("fii->f", grad_f)
    return coef_f[:, None] * (
        bm.einsum("fji,fj->fi", grad_f, face_normal)
        - (2.0 / 3.0) * div_u[:, None] * face_normal
    )
