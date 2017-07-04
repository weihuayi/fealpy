import numpy as np

def bc_to_point(point, entity, bc):
    pass

def grad_lambda(point, cell):
    NC = cell.shape[0]
    v0 = point[cell[:, 2], :] - point[cell[:, 1], :]
    v1 = point[cell[:, 0], :] - point[cell[:, 2], :]
    v2 = point[cell[:, 1], :] - point[cell[:, 0], :]
    dim = point.shape[1] 
    nv = np.cross(v2, -v1)
    Dlambda = np.zeros((NC, 3, dim), dtype=np.float)
    if dim == 2:
        length = nv
        W = np.array([[0, 1], [-1, 0]], dtype=np.float)
        Dlambda[:, 0, :] = v0@W/length.reshape(-1, 1)
        Dlambda[:, 1, :] = v1@W/length.reshape(-1, 1)
        Dlambda[:, 2, :] = v2@W/length.reshape(-1, 1)
        area = length/2.0
    elif dim == 3:
        length = np.sqrt(np.square(nv).sum(axis=1))
        n = nv/length.reshape(-1, 1)
        Dlambda[:, 0, :] = np.cross(n, v0)/length.reshape(-1,1)
        Dlambda[:, 1, :] = np.cross(n, v1)/length.reshape(-1,1)
        Dlambda[:, 2, :] = np.cross(n, v2)/length.reshape(-1,1)
        area = length/2.0
    return Dlambda, area



