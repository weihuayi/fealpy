
__all__ = [
    'random_gaussian_2d',
    'triangle_union_levelset',
    'random_ccw_triangles',
    'circle_union_levelset',
    'random_circles',
]

from typing import NamedTuple
import numpy as np
from fealpy.backend import bm, TensorLike as Tensor


# ------------------------------
#   Gaussian
# ------------------------------

Gaussian = NamedTuple("Gaussian", [("centers", Tensor), ("invcov", "Tensor")])


def random_gaussian_2d(
    num: int,
    box: tuple[float, float, float, float],
    major_lim: tuple[float, float],
    ecc_lim: tuple[float, float] = (0.5, 1.0),
):
    """Generate random means and covariance matrices for Gaussian distibution.

    Parameters:
        num (int): Number of i.i.d
        box (tuple of 4 floats): Limits for centers
        major_lim (tuple of 2 floats): Limits for the standard deviations
            on their major axis
        ecc_lim (tuple of 2 floats): Limits for eccentricity

    Returns:
        namedtuple:
        - Tensor: centers
        - Tensor: inverse of covariance matrices
    """
    theta = np.random.uniform(0, 2 * np.pi, size=(num,))
    major = np.random.uniform(major_lim[0], major_lim[1], size=(num,))
    ecc = np.random.uniform(ecc_lim[0], ecc_lim[1], size=(num,))
    ecc2 = ecc*ecc
    cos = np.cos(theta)
    sin = np.sin(theta)
    invcov = np.stack([
        np.stack([1-ecc2*sin**2, -ecc2*sin*cos], axis=-1),
        np.stack([-ecc2*sin*cos, 1-ecc2*cos**2], axis=-1),
    ], axis=-2) * (major**2)[:, None, None] # (NGuassian, GD, GD)
    centers = np.stack([
        np.random.uniform(box[0], box[1], size=(num,)),
        np.random.uniform(box[2], box[3], size=(num,)),
    ], axis=-1) # (NGuassian, GD)
    return Gaussian(centers, invcov)


# ------------------------------
#   Triangles
# ------------------------------

def triangle_union_levelset(points: Tensor, triangles: Tensor):
    """多个三角形并集的水平集函数（内部为负）

    Parameters:
        triangles (Tensor): shape (num, 3, 2) 逆时针排列的三角形顶点坐标
        points (Tensor): shape (N, 2) 待评估点坐标

    Returns:
        phi (ndarray): shape (N,) 并集的水平集函数值
    """
    # triangles: (num, 3, 2)
    # points: (N, 2)
    struct = points.shape[:-1]
    points = points.reshape(-1, points.shape[-1])

    # 边向量 e_k = v_{k+1} - v_k
    edges = bm.roll(triangles, -1, axis=1) - triangles  # (num, 3, 2)

    # 外法向量 n = (dy, -dx)
    normals = bm.stack((edges[..., 1], -edges[..., 0]), axis=-1)  # (num, 3, 2)

    # (N, num, 3, 2) = points[:,None,None,:] - triangles[None,:,:,:]
    diff = points[:, None, None, :] - triangles[None, :, :, :]

    # 点积 (x - v_k) · n_k → (N, num, 3)
    vals = bm.sum(diff * normals[None, :, :, :], axis=-1)

    # 每个三角形取 max → (N, num)
    phi_tri = bm.max(vals, axis=-1)

    # 并集取 min → (N,)
    phi = bm.min(phi_tri, axis=1)

    return phi.reshape(struct)


def random_ccw_triangles(num: int, /, xlim: tuple[float, float], ylim: tuple[float, float]):
    """随机生成满足逆时针排列的三角形

    Parameters:
        num (int): 三角形数量
        xlim (tuple[xmin, xmax]): x 坐标范围
        ylim (tuple[ymin, ymax]): y 坐标范围

    Returns:
        ndarray: 逆时针排列的三角形顶点 (num, 3, 2)
    """
    # 随机生成顶点
    triangles = np.empty((num, 3, 2), dtype=np.float64)
    triangles[..., 0] = np.random.uniform(xlim[0], xlim[1], size=(num, 3))
    triangles[..., 1] = np.random.uniform(ylim[0], ylim[1], size=(num, 3))

    # 叉积符号判断
    v0 = triangles[:, 0]
    v1 = triangles[:, 1]
    v2 = triangles[:, 2]

    cross = (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    )

    # 对顺时针的三角形交换 v1, v2
    mask = cross < 0
    triangles[mask, 1], triangles[mask, 2] = (
        np.copy(triangles[mask, 2]),
        np.copy(triangles[mask, 1]),
    )

    return bm.from_numpy(triangles)


def random_equ_triangles(num: int, /, xlim: tuple[float, float], ylim: tuple[float, float], rlim: tuple[float, float]):
    """随机生成逆时针排列的等边三角形

    Parameters:
        num (int): Number of triangles
        xlim (tuple[xmin, xmax]): 三角形中心 x 坐标范围
        ylim (tuple[ymin, ymax]): 三角形中心 y 坐标范围
        rlim (tuple[rmin, rmax]): 等边三角形外接圆半径范围（尺寸参数）

    Returns:
        ndarray: 逆时针排列的等边三角形顶点坐标 (num, 3, 2)
    """
    # 随机生成尺寸（外接圆半径）
    r = np.random.uniform(rlim[0], rlim[1], size=num)

    # 随机生成中心点
    centers = np.empty((num, 2), dtype=np.float64)
    centers[:, 0] = np.random.uniform(xlim[0]+r, xlim[1]-r, size=num)
    centers[:, 1] = np.random.uniform(ylim[0]+r, ylim[1]-r, size=num)

    # 随机初始角度
    theta0 = np.random.uniform(0.0, 2.0 * np.pi, size=num)

    # 三个顶点角度（逆时针）
    angles = theta0[:, None] + np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])

    # 生成顶点坐标
    triangles = np.empty((num, 3, 2), dtype=np.float64)
    triangles[..., 0] = centers[:, 0, None] + r[:, None] * np.cos(angles)
    triangles[..., 1] = centers[:, 1, None] + r[:, None] * np.sin(angles)

    return triangles


# ------------------------------
#   Circles
# ------------------------------

Circles = NamedTuple("Circles", [("centers", Tensor), ("radius", Tensor)])

def circle_union_levelset(p: Tensor, circles: tuple[Tensor, Tensor]):
    """Calculate level set function value.

    Parameters:
        p (Tensor): points (NN, 2) to be evaluated
        circles (tuple of two tensors): centers (NC, 2) and radius (NC,)

    Returns:
        value of the levelset function on the given points, shaped (NN,).
    """
    centers, radius = circles
    struct = p.shape[:-1]
    p = p.reshape(-1, p.shape[-1])
    dis = bm.linalg.norm(p[:, None, :] - centers[None, :, :], axis=-1) # (N, NCir)
    ret = bm.min(dis - radius[None, :], axis=-1) # (N, )
    return ret.reshape(struct)


def random_circles(num: int, /) -> Circles:
    """Generate random circles on [-1, 1]^2"""
    ctrs_ = np.random.rand(num, 2) * 1.6 - 0.8 # (NCir, GD)
    b = np.min(0.9-np.abs(ctrs_), axis=-1) # (NCir, )
    rads_ = np.random.rand(num) * (b-0.1) + 0.1 # (NCir, )
    return Circles(bm.from_numpy(ctrs_), bm.from_numpy(rads_))
