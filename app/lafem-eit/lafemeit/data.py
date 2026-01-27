
__all__ = [
    'random_gaussian2d_model',
    'random_unioned_circles_model',
    'random_unioned_triangles_model',
]

from typing import Literal
from collections import namedtuple

from fealpy.backend import bm, TensorLike as Tensor

from . import shapes as _S


FunctionModel = namedtuple("FunctionModel", ["coef", "levelset", "shape"])


def _gaussian(p: Tensor, centers: Tensor, inv_cov: Tensor, base: float = 10.):
    # p (NC, NQ, GD)
    # centers (NGuassian, GD)
    # inv_cov (NGuassian, GD, GD)
    struct = p.shape[:-1]
    p = p.reshape(-1, p.shape[-1])
    p0 = p[:, None, :] - centers[None, :, :] # (NC*NQ, NGuassian, GD)
    ind = bm.exp(-0.5 * bm.einsum("ngi, gij, ngj -> ng", p0, inv_cov, p0))
    ind = bm.sum(ind, axis=-1).reshape(struct) # (NC, NQ)
    minim = bm.min(ind)
    maxim = bm.max(ind)
    ind = (ind - minim) / (maxim - minim)
    return base**ind


def random_gaussian2d_model(
    num: int,
    box: tuple[float, float, float, float],
    major_lim: tuple[float, float],
    ecc_lim: tuple[float, float] = (0.5, 1.0),
    base: float = 10.
):
    """
    Generate a random Gaussian conductivity model.

    Parameters:
        num (int): Number of Gaussian
        box (tuple[xmin, xmax, ymin, ymax]): The box of Gaussian centers
        major_lim (tuple[major_min, major_max]): The major axis covariance range
        ecc_lim (tuple[ecc_min, ecc_max]): The eccentricity range
        base (float): The ratio of the conductivity value that ranges [1.0, ratio]

    Returns:
        namedtuple
        - coef (Callable): A function that takes points and returns the conductivity values.
        - levelset (None): No levelset function.
    """
    gaussian = _S.random_gaussian_2d(
        num, box, major_lim, ecc_lim
    )

    def gaussian_conductivity(points: Tensor, *args, **kwargs) -> Tensor:
        return _gaussian(
            points,
            bm.from_numpy(gaussian.centers),
            bm.from_numpy(gaussian.invcov),
            base
        )

    gaussian_conductivity.__dict__['coordtype'] = 'cartesian'

    return FunctionModel(gaussian_conductivity, None, gaussian)


def random_unioned_circles_model(
    num: int,
    values: tuple[float, float] = (10.0, 1.0),
):
    """Generate a random unioned circles conductivity model.

    Parameters:
        num (int): Number of circles
        values (tuple[float, float]): The conductivity values of the circles and the rest.

    Returns:
        namedtuple:
        - coef (Callable): A function that takes points and returns the conductivity values.
        - levelset (Callable): Level set function.
    """
    circles = _S.random_circles(num)
    level_set_func = lambda p: _S.circle_union_levelset(p, circles)

    def circle_conductivity(points: Tensor, *args, **kwargs) -> Tensor:
        inclusion = level_set_func(points) < 0. # a bool tensor on quadrature points.
        sigma = bm.empty(points.shape[:2], **bm.context(points)) # (Q, C)
        sigma = bm.set_at(sigma, inclusion, values[0])
        sigma = bm.set_at(sigma, ~inclusion, values[1])
        return sigma

    circle_conductivity.__dict__['coordtype'] = 'cartesian'

    return FunctionModel(circle_conductivity, level_set_func, circles)


def random_unioned_triangles_model(
    num: int,
    box: tuple[float, float, float, float] = [-1., 1., -1., 1.],
    kind: Literal["", "equ"] = "",
    rlim: tuple[float, float] = (0.5, 1.0),
    values: tuple[float, float] = (10.0, 1.0),
):
    """Generate a random unioned triangles conductivity model.

    Parameters:
        num (int): Number of triangles
        box (tuple[xmin, xmax, ymin, ymax]): The box of triangles
        kind (str): The kind of triangles.
        rlim (tuple[r_min, r_max]): The radius range, available when kind is "equ".
        values (tuple[float, float]): The conductivity values of the triangles and the rest.

    Returns:
        namedtuple:
        - coef (Callable): A function that takes points and returns the conductivity values.
        - levelset (Callable): Level set function.
    """
    if kind == "equ":
        tris = _S.random_equ_triangles(num, box[0:2], box[2:4], rlim)
    else:
        tris = _S.random_ccw_triangles(num, box[0:2], box[2:4])
    level_set_func = lambda p: _S.triangle_union_levelset(p, tris)

    def triangle_conductivity(points: Tensor, *args, **kwargs) -> Tensor:
        inclusion = level_set_func(points) < 0. # a bool tensor on quadrature points.
        sigma = bm.empty(points.shape[:2], **bm.context(points)) # (Q, C)
        sigma = bm.set_at(sigma, inclusion, values[0])
        sigma = bm.set_at(sigma, ~inclusion, values[1])
        return sigma

    triangle_conductivity.__dict__['coordtype'] = 'cartesian'

    return FunctionModel(triangle_conductivity, level_set_func, tris)
