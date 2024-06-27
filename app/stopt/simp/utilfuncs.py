import numpy as np

def compute_filter(mesh, rmin):
    nelx, nely = mesh.nx, mesh.ny
    NC = mesh.number_of_cells()
    H = np.zeros((NC, NC))

    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = (i1) * nely + j1
            imin = max(i1 - (np.ceil(rmin) - 1), 0.)
            imax = min(i1 + (np.ceil(rmin)), nelx)
            for i2 in range(int(imin), int(imax)):
                jmin = max(j1 - (np.ceil(rmin) - 1), 0.)
                jmax = min(j1 + (np.ceil(rmin)), nely)
                for j2 in range(int(jmin), int(jmax)):
                    e2 = i2 * nely + j2
                    H[e1, e2] = max(0., rmin - \
                                    np.sqrt((i1 - i2)**2 + (j1-j2)**2))

    Hs = np.sum(H, 1)
    return H, Hs


def sensitivity_filter(ft, rho, dc, dv):
    """
    应用 mesh-independency filter 进行每个单元的灵敏度过滤.

    Parameters:
    - rmin (float): Filter 半径.
    - rho (ndarray - (nely, nelx) ): 密度分布矩阵.
    - dc (ndarray - (nely, nelx) ): 原始的灵敏度矩阵.

    Returns:
    - dcn(ndarray - (nely, nelx) ): 过滤后的灵敏度矩阵.
    """

    if (ft['type'] == 1):
        dc = np.matmul(ft['H'], \
                       np.multiply(rho, dc) / ft['Hs'] / np.maximum(1e-3, rho))
    elif (ft['type'] == 2):
        dc = np.matmul(ft['H'], (dc / ft['Hs']))
        dv = np.matmul(ft['H'], (dv / ft['Hs']))
    return dc, dv