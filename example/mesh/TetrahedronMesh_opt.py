import gmsh
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TetrahedronMesh
from fealpy.mesh import QuadrangleMesh

def quality_matrix(mesh):
    NN = mesh.number_of_nodes()
    NC = mesh.number_of_cells()

    node = mesh.entity('node')
    cell = mesh.entity('cell')

    v10 = node[cell[:, 0]] - node[cell[:, 1]]
    v20 = node[cell[:, 0]] - node[cell[:, 2]]
    v30 = node[cell[:, 0]] - node[cell[:, 3]]

    v21 = node[cell[:, 1]] - node[cell[:, 2]]
    v31 = node[cell[:, 1]] - node[cell[:, 3]]
    v32 = node[cell[:, 2]] - node[cell[:, 3]]

    l10 = np.sum(v10**2, axis=-1)
    l20 = np.sum(v20**2, axis=-1)
    l30 = np.sum(v30**2, axis=-1)
    l21 = np.sum(v21**2, axis=-1)
    l31 = np.sum(v31**2, axis=-1)
    l32 = np.sum(v32**2, axis=-1)

    d0 = np.zeros((NC, 3), dtype=mesh.ftype)
    c12 =  np.cross(v10, v20)
    d0 += l30[:, None]*c12
    c23 = np.cross(v20, v30)
    d0 += l10[:, None]*c23
    c31 = np.cross(v30, v10)
    d0 += l20[:, None]*c31

    c12 = np.sum(c12*d0, axis=-1)
    c23 = np.sum(c23*d0, axis=-1)
    c31 = np.sum(c31*d0, axis=-1)
    c = c12 + c23 + c31

    A = np.zeros((NC, 4, 4), dtype=mesh.ftype)
    A[:, 0, 0]  = 2*c
    A[:, 0, 1] -= 2*c23
    A[:, 0, 2] -= 2*c31
    A[:, 0, 3] -= 2*c12

    A[:, 1, 1] = 2*c23
    A[:, 2, 2] = 2*c31
    A[:, 3, 3] = 2*c12
    A[:, 1:, 0] = A[:, 0, 1:]

    K = np.zeros((NC, 4, 4), dtype=mesh.ftype)
    K[:, 0, 1] -= l30 - l20
    K[:, 0, 2] -= l10 - l30
    K[:, 0, 3] -= l20 - l10
    K[:, 1:, 0] -= K[:, 0, 1:]

    K[:, 1, 2] -= l30
    K[:, 1, 3] += l20
    K[:, 2:, 1] -= K[:, 1, 2:]

    K[:, 2, 3] -= l10
    K[:, 3, 2] += l10

    S = np.zeros((NC, 4, 4), dtype=mesh.ftype)
    fm = mesh.entity_measure("face")
    cm = mesh.entity_measure("cell")
    c2f = mesh.ds.cell_to_face()

    s = fm[c2f]
    s_sum = np.sum(s, axis=-1)
     
    p0 = (l31/s[:,2] + l21/s[:,3] + l32/s[:,1])/4
    p1 = (l32/s[:,0] + l20/s[:,3] + l30/s[:,2])/4
    p2 = (l30/s[:,1] + l10/s[:,3] + l31/s[:,0])/4
    p3 = (l10/s[:,2] + l20/s[:,1] + l21/s[:,0])/4

    q10 = -(np.sum(v31*v30, axis=-1)/s[:,2]+np.sum(v21*v20, axis=-1)/s[:,3])/4
    q20 = -(np.sum(v32*v30, axis=-1)/s[:,1]+np.sum(-v21*v10, axis=-1)/s[:,3])/4
    q30 = -(np.sum(-v32*v20, axis=-1)/s[:,1]+np.sum(-v31*v10, axis=-1)/s[:,2])/4
    q21 = -(np.sum(v32*v31, axis=-1)/s[:,0]+np.sum(v20*v10, axis=-1)/s[:,3])/4
    q31 = -(np.sum(v30*v10, axis=-1)/s[:,2]+np.sum(-v32*v21, axis=-1)/s[:,0])/4
    q32 = -(np.sum(v31*v21, axis=-1)/s[:,0]+np.sum(v30*v20, axis=-1)/s[:,1])/4
    
    S[:, 0, 0] = p0
    S[:, 0, 1] = q10
    S[:, 0, 2] = q20
    S[:, 0, 3] = q30
    S[:, 1:,0] = S[:, 0, 1:]

    S[:, 1, 1] = p1
    S[:, 1, 2] = q21
    S[:, 1, 3] = q31
    S[:, 2:,1] = S[:, 1, 2:]

    S[:, 2, 2] = p2
    S[:, 2, 3] = q32
    S[:, 3, 2] = q32
    S[:, 3, 3] = p3
    
    C0 = np.zeros((NC, 4, 4), dtype=np.float_)
    C1 = np.zeros((NC, 4, 4), dtype=np.float_)
    C2 = np.zeros((NC, 4, 4), dtype=np.float_)

    def f(CC, xx):
        CC[:, 0, 1] = xx[:, 2]
        CC[:, 0, 2] = xx[:, 3]
        CC[:, 0, 3] = xx[:, 1]
        CC[:, 1, 0] = xx[:, 3]
        CC[:, 1, 2] = xx[:, 0]
        CC[:, 1, 3] = xx[:, 2]
        CC[:, 2, 0] = xx[:, 1]
        CC[:, 2, 1] = xx[:, 3]
        CC[:, 2, 3] = xx[:, 0]
        CC[:, 3, 0] = xx[:, 2]
        CC[:, 3, 1] = xx[:, 0]
        CC[:, 3, 2] = xx[:, 1]

    f(C0, node[cell, 0])
    f(C1, node[cell, 1])
    f(C2, node[cell, 2])

    C0 = 0.5*(-C0 + C0.swapaxes(-1, -2))
    C1 = 0.5*(C1  - C1.swapaxes(-1, -2))
    C2 = 0.5*(-C2 + C2.swapaxes(-1, -2))
    return A,K,d0

mesh = TetrahedronMesh.one_tetrahedron_mesh()
A,K,d0 = quality_matrix(mesh)
print(A)
print(K)
print(d0)


# Create the 3D plot
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
#mesh.add_plot(axes, threshold=lambda p: p[..., 0] > 0.0)
mesh.add_plot(axes)
plt.show()





