import jax
import jax.numpy as jnp

def value_and_jacfwd(f, x):
    pushfwd = functools.partial(jax.jvp, f, (x, ))
    basis = jnp.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
    y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis, ))
    return y, jac

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
