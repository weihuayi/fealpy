"""Lid-driven cavity data adapter for collocated Navier-Stokes FVM solvers."""

from typing import Sequence

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesher import BoxMesher2d


class LidDrivenCavityCase(BoxMesher2d):
    """PDE-like case object for the 2D lid-driven cavity benchmark.

    The current collocated SIMPLE/PISO solvers consume objects that look like
    entries from ``fealpy.model.navier_stokes``.  This adapter exposes the same
    small method set without providing manufactured exact solutions.

    Notes
    -----
    ``rho`` and ``mu`` define the momentum-equation coefficients used by
    ``NSFVMSimpleModel`` and ``NSFVMPISOModel``.  If ``mu`` is omitted, it is
    derived from ``re = rho * |U_lid| * L / mu``.
    """

    default_mesh_type = "uniform_quad"
    supports_geometric_refine = False

    def __init__(
        self,
        re: float = 1.0,
        *,
        domain: Sequence[float] = (0.0, 1.0, 0.0, 1.0),
        lid_velocity: float = 1.0,
        rho: float = 1.0,
        mu: float | None = None,
        eps: float = 1.0e-12,
    ) -> None:
        if re <= 0.0:
            raise ValueError("re must be positive.")
        if rho <= 0.0:
            raise ValueError("rho must be positive.")
        if len(domain) != 4:
            raise ValueError("domain must be (xmin, xmax, ymin, ymax).")
        self.box = [float(value) for value in domain]
        length = self.box[1] - self.box[0]
        self.rho = float(rho)
        self.lid_velocity = float(lid_velocity)
        self.re = float(re)
        if mu is None:
            self.mu = self.rho * abs(self.lid_velocity) * length / self.re
        else:
            if mu <= 0.0:
                raise ValueError("mu must be positive.")
            self.mu = float(mu)
            if abs(self.lid_velocity) * length > 0.0:
                self.re = self.rho * abs(self.lid_velocity) * length / self.mu
        self.nu = self.mu / self.rho
        self.eps = float(eps)
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box

    @cartesian
    def source(self, p: TensorLike, t: float | None = None) -> TensorLike:
        return bm.zeros(
            p.shape,
            dtype=p.dtype,
            device=bm.get_device(p),
        )

    @cartesian
    def dirichlet_velocity(self, p: TensorLike) -> TensorLike:
        y = p[..., 1]
        top = bm.abs(y - self.box[3]) <= self.eps
        u = bm.where(top, self.lid_velocity * bm.ones_like(y), bm.zeros_like(y))
        v = bm.zeros_like(y)
        return bm.stack([u, v], axis=-1)

    @cartesian
    @cartesian
    def velocity_0(self, p: TensorLike, t: float = 0.0) -> TensorLike:
        return bm.zeros(
            p.shape,
            dtype=p.dtype,
            device=bm.get_device(p),
        )

    @cartesian
    def pressure_0(self, p: TensorLike, t: float = 0.0) -> TensorLike:
        return bm.zeros(
            p.shape[:-1],
            dtype=p.dtype,
            device=bm.get_device(p),
        )
