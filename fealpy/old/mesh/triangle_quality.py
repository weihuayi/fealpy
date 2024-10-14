import numpy as np

def radius_ratio(mesh):
    node = mesh.node
    cell = mesh.ds.cell
    NC = mesh.number_of_cells()

    localEdge = mesh.ds.local_edge()
    v = [node[cell[:, j], :] - node[cell[:, i], :] for i, j in localEdge]
    J = np.zeros((NC,2,2))
    J[:,0]=v[2]
    J[:,1]=-v[1]
    detJ = np.linalg.det(J)
    l2 = np.zeros((NC, 3))
    for i in range(3):
        l2[:, i] = np.sum(v[i]**2, axis=1)
    l = np.sqrt(l2)
    p = l.sum(axis=1)
    q = l.prod(axis=1)
    area = np.cross(v[1], v[2])/2
    quality = (p*q)/(16*area**2)
    return 1/quality
