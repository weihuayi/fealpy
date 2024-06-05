import numpy as np
from scipy.optimize import fsolve

def project_point_on_sphere_to_implict_surface(nodes, center, fun):
    """
    @brief : Project the points `nodes` on the sphere to the implicit surface defined by `fun`.
    @param nodes : The points on the sphere.
    @param center : The center of the sphere.
    @param fun : The implicit surface function.
    """
    ret = np.zeros_like(nodes)
    for i, node in enumerate(nodes):
        g = lambda t : fun(center + t*(node-center))
        t = fsolve(g, 1000)
        ret[i] = center + t*(node-center)
    return ret

def project_point_to_sphere(nodes, center, radius):
    """
    @brief : Project the points `nodes` to the sphere defined by `center` and `radius`.
    @param nodes : The points to be projected.
    @param center : The center of the sphere.
    @param radius : The radius of the sphere.
    """
    v = nodes-center[None, :]
    ret = center + radius*v/np.linalg.norm(v)
    return ret




