
import numpy as np

def error(mesh, u, v, power=2, celltype=False, q=None):
    """

    @brief 给定两个函数，计算两个函数的之间的差，默认计算 L2 差（power=2)
           power 的取值可以是任意的 p

    TODO
    ----
    1. 考虑无穷范数的情形
    """
    mesh = self.mesh
    GD = mesh.geo_dimension()

    qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    ps = mesh.bc_to_point(bcs)

    if callable(u):
        if not hasattr(u, 'coordtype'): 
            u = u(ps)
        else:
            if u.coordtype == 'cartesian':
                u = u(ps)
            elif u.coordtype == 'barycentric':
                u = u(bcs)

    if callable(v):
        if not hasattr(v, 'coordtype'):
            v = v(ps)
        else:
            if v.coordtype == 'cartesian':
                v = v(ps)
            elif v.coordtype == 'barycentric':
                v = v(bcs)

    if u.shape[-1] == 1:
        u = u[..., 0]

    if v.shape[-1] == 1:
        v = v[..., 0]

    f = np.power(np.abs(u - v), power) 
    if isinstance(f, (int, float)): # f为标量常函数
        e = f*self.cellmeasure
    elif isinstance(f, np.ndarray):
        if f.shape == (GD, ): # 常向量函数
            e = self.cellmeasure[:, None]*f
        elif f.shape == (GD, GD):
            e = self.cellmeasure[:, None, None]*f
        else:
            e = np.einsum('q, qc..., c->c...', ws, f, self.cellmeasure)

    if celltype == False:
        e = np.power(np.sum(e), 1/power)
    else:
        e = np.power(np.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
    return e # float or (NC, )
