import numpy as np
from functools import partial
import jax
import jax.numpy as jnp

def value_and_jacfwd(f, x):
    pushfwd = partial(jax.jvp, f, (x, ))
    basis = jnp.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
    y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis, ))
    return y, jac

# simplex 
def _simplex_shape_function(bc, mi, p):
    """
    @brief 给定一组重心坐标点 `bc`, 计算单纯形单元上 `p` 次 Lagrange
    基函数在这一组重心坐标点处的函数值

    @param[in] bc : (TD+1, )
    @param[in] p : 基函数的次数，为正整数
    @param[in] mi : p 次的多重指标矩阵

    @return phi : 形状为 (NQ, ldof)
    """
    TD = bc.shape[-1] - 1
    c = jnp.arange(1, p+1)
    P = 1.0/jnp.cumprod(c)
    t = jnp.arange(0, p)
    A = p*bc - jnp.arange(0, p).reshape(-1, 1)
    A = P.reshape(-1, 1)*jnp.cumprod(A, axis=-2) # (p, TD+1)
    B = jnp.ones((p+1, TD+1), dtype=A.dtype)
    B = B.at[1:, :].set(A)
    idx = jnp.arange(TD+1)
    phi = jnp.prod(B[mi, idx], axis=-1)
    return phi

def _diff_simplex_shape_function(bc, mi, p, n):
    fn = _simplex_shape_function
    for i in range(n):
        fn = jax.jacfwd(fn)
    return fn(bc, mi, p)

@partial(jax.jit, static_argnums=(2, ))
def simplex_shape_function(bcs, mi, p):
    fn = jax.vmap(_simplex_shape_function, in_axes=(0, None, None))
    return fn(bcs, mi, p)

@partial(jax.jit, static_argnums=(2, 3))
def diff_simplex_shape_function(bcs, mi, p, n): 
    return jax.vmap(
            _diff_simplex_shape_function, 
            in_axes=(0, None, None, None)
            )(bcs, mi, p, n)

# edge 
def edge_length(points):
    return jnp.linalg.norm(points[1] - points[0])

@partial(jax.jit, static_argnums=(2, ))
def edge_to_ipoint(edges, indices, p):
    NN = jnp.max(edges[:, ])+1
    return jnp.hstack([
        edges[:, 0].reshape(-1, 1), (p-1)*indices.reshape(-1, 1) +
        jnp.arange(p-1)+ NN , edges[:, 1].reshape(-1, 1)])


# triangle 
def tri_area_2d(points):
    """
    @brief 给定一个单元的三个顶点的坐标，计算三角形的面积
    """
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    nv = jnp.cross(v1, v2)
    return nv/2.0

def tri_area_3d(points):
    """
    @brief 给定一个单元的三个顶点的坐标，计算三角形的面积

    @params points : (3, 3) 
    """
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    nv = jnp.cross(v1, v2)
    a = jnp.norm(nv)/2.0
    return nv/2.0

def tri_area_2d_with_jac(points):
    return value_and_jacfwd(tri_area_2d, points)

def tri_area_3d_with_jac(points):
    return value_and_jacfwd(tri_area_3d, points)

def tri_quality_radius_ratio(points):
    v0 = points[2] - points[1]
    v1 = points[0] - points[2]
    v2 = points[1] - points[0]

    l0 = jnp.linalg.norm(v0)
    l1 = jnp.linalg.norm(v1)
    l2 = jnp.linalg.norm(v2)

    p = l0 + l1 + l2
    q = l0*l1*l2
    nv = np.cross(v1, v2)
    a = jnp.norm(nv)/2.0
    quality = p*q/(16*a**2)
    return quality

def tri_quality_radius_ratio_with_jac(points):
    return value_and_jacfwd(tri_quality_radius_ratio, points)

def tri_grad_lambda_2d(points):
    """
    @brief 计算2D三角形单元的形函数梯度 

    @params points : 形状为  (3, 2), 存储一个三角形单元的坐标，逆时针方向
    """
    v0 = points[2] - points[1]
    v1 = points[0] - points[2]
    v2 = points[1] - points[0]
    nv = jnp.cross(v1, v2) # 三角形面积的 2 倍 
    Dlambda = jnp.array([
        [-v0[1], v0[0]], 
        [-v1[1], v1[0]], 
        [-v2[1], v2[0]]], dtype=jnp.float64)/nv
    return Dlambda 

def tri_grad_lambda_3d(points):
    """
    @brief 计算 3D 三角形单元的形函数梯度 

    @params points : 形状为  (3, 3), 存储一个三角形单元的坐标(逆时针方向)
    """
    v0 = points[2] - points[1]
    v1 = points[0] - points[2]
    v2 = points[1] - points[0]
    nv = jnp.cross(v1, v2) # 三角形面积的 2 倍 
    length = jnp.linalg.norm(nv)
    n = nv/length
    n0 = jnp.cross(n, v0)
    n1 = jnp.cross(n, v1)
    n2 = jnp.cross(n, v2)
    Dlambda = jnp.array([n0, n1, n2], dtype=jnp.float64)/length
    return Dlambda 


# tetrahedron
