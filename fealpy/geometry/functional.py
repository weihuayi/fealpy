
from typing import Callable
from math import sqrt

from .. import logger
from ..backend import backend_manager as bm
from ..backend import TensorLike


def apply_rotation(
        points: TensorLike, 
        centers: TensorLike, 
        rotation: TensorLike, GD: int) -> TensorLike:
    """Apply rotation to points relative to given centers.

    Parameters:
        points (TensorLike): Points to rotate, shape (NP, GD).
        centers (TensorLike): Centers of rotation, shape (NC, GD).
        rotation (TensorLike): Rotation angles in radians.
            - For 2D: shape (NC, 1).
            - For 3D: shape (NC, 3).
        GD (int): Geometric dimension (2 or 3).

    Returns:
        TensorLike: Rotated points.
    """
    if GD == 2:
        # For 2D, apply rotation around the center
        angle = rotation[:, 0]  # Rotation angle in radians
        cos_angle = bm.cos(angle)
        sin_angle = bm.sin(angle)

        translated_points = points - centers[:, None, :]
        rotated_points = bm.stack([
            cos_angle[:, None] * translated_points[..., 0] - sin_angle[:, None] * translated_points[..., 1],
            sin_angle[:, None] * translated_points[..., 0] + cos_angle[:, None] * translated_points[..., 1]
        ], axis=-1)
        return rotated_points + centers[:, None, :]

    elif GD == 3:
        # For 3D, apply rotation around each axis (assuming rotation order is x, y, z)
        angles = rotation
        translated_points = points - centers[:, None, :]

        # Rotation around x-axis
        cos_angle_x = bm.cos(angles[:, 0])[:, None]
        sin_angle_x = bm.sin(angles[:, 0])[:, None]
        rotated_points = bm.stack([
            translated_points[..., 0],
            cos_angle_x * translated_points[..., 1] - sin_angle_x * translated_points[..., 2],
            sin_angle_x * translated_points[..., 1] + cos_angle_x * translated_points[..., 2]
        ], axis=-1)

        # Rotation around y-axis
        cos_angle_y = bm.cos(angles[:, 1])[:, None]
        sin_angle_y = bm.sin(angles[:, 1])[:, None]
        rotated_points = bm.stack([
            cos_angle_y * rotated_points[..., 0] + sin_angle_y * rotated_points[..., 2],
            rotated_points[..., 1],
            -sin_angle_y * rotated_points[..., 0] + cos_angle_y * rotated_points[..., 2]
        ], axis=-1)

        # Rotation around z-axis
        cos_angle_z = bm.cos(angles[:, 2])[:, None]
        sin_angle_z = bm.sin(angles[:, 2])[:, None]
        rotated_points = bm.stack([
            cos_angle_z * rotated_points[..., 0] - sin_angle_z * rotated_points[..., 1],
            sin_angle_z * rotated_points[..., 0] + cos_angle_z * rotated_points[..., 1],
            rotated_points[..., 2]
        ], axis=-1)

        return rotated_points + centers[:, None, :]


def msign(x: TensorLike, eps=1e-10) -> TensorLike:
    flag = bm.sign(x)
    return bm.set_at(flag, bm.abs(x) < eps, 0)


def find_cut_point(phi: Callable, p0: TensorLike, p1: TensorLike) -> TensorLike:
    """Find the zero-cross point of the curve on the line segment.
    Assume that all the edges provided are cut by the curve."""
    if bm.backend_name == "jax":
        logger.warning("`find_cut_point` is tested to have low performance on JAX backend, "
                       "Avoid to use it in the main loop if you encounter performance issue.")

    set_at = bm.set_at
    nonzero = bm.nonzero
    NUM = p0.shape[0]
    pl = bm.copy(p0)
    pr = bm.copy(p1)
    pc = bm.empty_like(p0) # point cut
    phil = phi(p0)
    phir = phi(p1)
    phic = bm.empty_like(phil)

    h = bm.linalg.norm(p1 - p0, axis=1)
    EPS = bm.finfo(p0.dtype).eps
    tol = sqrt(EPS) * h * h
    flag = bm.arange(NUM, dtype=bm.int32) # 需要调整的边.

    while flag.shape[0] > 0:
        # evaluate the sign.
        pc = set_at(pc, flag, (pl[flag] + pr[flag]) / 2.)
        phic = set_at(phic, flag, phi(pc[flag]))
        cphic = phic[flag]
        left_idx = nonzero(phil[flag] * cphic > 0)[0]
        right_idx = nonzero(phir[flag] * cphic > 0)[0]

        # move the point.
        pl = set_at(pl, left_idx, pc[left_idx])
        pr = set_at(pr, right_idx, pc[right_idx])
        phil = set_at(phil, left_idx, phic[left_idx])
        phir = set_at(phir, right_idx, phic[right_idx])
        h = set_at(h, slice(None), h/2.)

        continue_signal = (h[flag] > tol[flag]) & (phic[flag] != 0)
        flag = flag[continue_signal]

    return pc


def project(imfun, p0:TensorLike, maxit=200, tol=1e-13, returngrad=False, returnd=False):

    eps = bm.finfo(float).eps
    p = p0
    value = imfun(p)
    s = bm.sign(value)
    grad = imfun.gradient(p)
    lg = bm.sum(grad**2, axis=-1, keepdims=True)
    grad /= lg
    grad *= value[..., None]
    pp = p - grad
    v = s[..., None]*(pp - p0)
    d = bm.sqrt(bm.sum(v**2, axis=-1, keepdims=True))
    d *= s[..., None]

    g = imfun.gradient(pp)
    g /= bm.sqrt(bm.sum(g**2, axis=-1, keepdims=True))
    g *= d
    p = p0 - g

    k = 0
    while True:
        value = imfun(p)
        grad = imfun.gradient(p)
        lg = bm.sqrt(bm.sum(grad**2, axis=-1, keepdims=True))
        grad /= lg

        v = s[..., None]*(p0 - p)
        d = bm.sqrt(bm.sum(v**2, axis=-1))
        isOK = d < eps
        d[isOK] = 0
        v[isOK] = grad[isOK]
        v[~isOK] /= d[~isOK][..., None]
        d *= s

        ev = grad - v
        e = bm.max(bm.sqrt((value/lg.reshape(lg.shape[0:-1]))**2 + bm.sum(ev**2, axis=-1)))
        if e < tol:
            break
        else:
            k += 1
            if k > maxit:
                break
            grad /= lg
            grad *= value[..., None]
            pp = p - grad
            v = s[..., None]*(pp - p0)
            d = bm.sqrt(bm.sum(v**2, axis=-1, keepdims=True))
            d *= s[..., None]

            g = imfun.gradient(pp)
            g /= bm.sqrt(bm.sum(g**2, axis=-1, keepdims=True))
            g *= d
            p = p0 - g

    if (returnd is True) and (returngrad is True):
        return p, d, grad
    elif (returnd is False) and (returngrad is True):
        return p, grad
    elif (returnd is True) and (returngrad is False):
        return p, d
    else:
        return p
