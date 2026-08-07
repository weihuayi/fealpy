r"""Cell-average exact values and errors for finite-volume solutions.

This helper intentionally does not call ``mesh.error(exact, numerical)``
directly.  The mesh-level error routine treats ``numerical`` as a
piecewise-constant P0 function and computes the continuous L2 error

.. math::
    \left(\sum_K\int_K |u(x)-u_{h,K}|^2\,dx\right)^{1/2}.

That is a valid P0 reconstruction error, but it includes the unavoidable
within-cell variation of the exact function.  Even if ``u_{h,K}`` is the exact
control-volume average of a smooth solution, ``u(x)-u_{h,K}`` is generally
O(h) inside each cell, so this continuous P0 L2 error typically decreases only
at first order.

The cell-centred FVM unknown is compared here with the exact control-volume
average

.. math::
    \bar u_K = |K|^{-1}\int_K u(x)\,dx,

and the reported norm is

.. math::
    \left(\sum_K |K|\,|u_{h,K}-\bar u_K|^2\right)^{1/2}.

This removes the P0 representation error and measures the finite-volume cell
average error.  For the current Poisson FVM examples this is the quantity that
shows the expected second-order convergence.
"""

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .fvm_geometry import FVMGeometry


def cell_average(mesh, func, *, q: int = 4, geometry=None) -> TensorLike:
    r"""Return the control-volume average of an exact function.

    Finite-volume unknowns represent cell averages.  For an exact function
    ``u`` this routine computes

    .. math::
        \bar u_K = \frac{1}{|K|}\int_K u(x)\,dx

    on every cell ``K``.
    """
    geometry = FVMGeometry(mesh) if geometry is None else geometry
    integral = geometry.cell_integral(
        lambda points, _: func(points), q=q
    )
    cell_measure = geometry.cell_measure
    if integral.ndim == 1:
        return integral / cell_measure
    return integral / cell_measure.reshape((cell_measure.shape[0],) + (1,) * (integral.ndim - 1))


def cell_average_l2_error(
    mesh,
    func,
    numerical: TensorLike,
    *,
    q: int = 4,
    geometry=None,
):
    r"""Return the discrete L2 error against exact control-volume averages.

    ``mesh.error(func, numerical)`` is not used here because it measures the
    continuous L2 error of the P0 reconstruction.  For smooth exact solutions
    that error is dominated by the variation of ``func`` within each cell and
    is usually first order.  This routine first computes the exact cell
    averages and then compares those averages with the FVM unknowns, which is
    the error quantity used to observe second-order convergence of the
    cell-average solution.
    """
    geometry = FVMGeometry(mesh) if geometry is None else geometry
    average = cell_average(mesh, func, q=q, geometry=geometry)
    cell_measure = geometry.cell_measure
    diff = numerical - average
    if diff.ndim == 1:
        error = bm.sqrt(bm.sum(cell_measure * diff**2))
    else:
        error = bm.sqrt(bm.sum(cell_measure[:, None] * diff**2, axis=0))
    return error, average
