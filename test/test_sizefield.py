import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, spdiags, eye, bmat

from fealpy.mesh import UniformMesh2d, MeshFactory, QuadrangleMesh, TriangleMesh
import time

def ff(x):
    y = np.linalg.norm(x - np.array([[-0.1, -0.1]]), axis=-1)
    #return np.sin(np.pi*x[..., 0])*np.sin(np.pi*x[..., 1])
    return np.sin(2*np.pi*x[..., 0])*np.sin(2*np.pi*x[..., 1])+1
    #return y**2

def exu(x):
    return np.sin(np.pi*x[..., 0])*np.sin(np.pi*x[..., 1])

def source(x):
    return 2*np.pi**2*exu(x)

def t2b(mesht, meshb, tf):
    '''!
    @param f 一维数组，长度为 mesht 中的节点数
    '''
    origin = meshb.origin
    h = meshb.h
    tnode = mesht.entity('node')

    Xp = (tnode-origin)/h # (NN, 2)
    base = (Xp - 0.5).astype(np.int_)
    fx = Xp - base # (NN, 2)
    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

    bf = meshb.function()
    numf = meshb.function()
    for i in range(3):
        for j in range(3):
            weight = w[i][:, 0] * w[j][:, 1] # (NN, )
            for k in range(len(tnode)):
                bf[base[k, 0]+i, base[k, 1]+j] += weight[k] * tf[k]
                numf[base[k, 0]+i, base[k, 1]+j] += weight[k]
    flag = numf > 0
    bf[flag] = bf[flag]/numf[flag] # 加权平均

    # 背景网格太密
    if np.any(~flag):
        print('背景网格太密')
        x = bf.reshape(-1)
        pnode = meshb.entity('node').reshape(-1, 2)

        NN = meshb.number_of_nodes()
        F = np.zeros(NN, dtype=np.float_)
        A = meshb.stiff_matrix()
        isDDof = flag.flat 

        # 边界条件处理
        F -= A@x
        bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        bdIdx[isDDof] = 1
        Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd
        F[isDDof] = x[isDDof]

        x = spsolve(A, F)
        bf = x.reshape(bf.shape)

        if 0:
            # 画图
            node0 = np.c_[pnode, x[:, None]]
            node1 = np.c_[pnode, ff(pnode)[:, None]]
            cell0 = meshb.entity('cell')[:, [0, 2, 1]]
            cell1 = meshb.entity('cell')[:, [1, 2, 3]]
            cell = np.r_[cell0, cell1]

            mesh0 = TriangleMesh(node0, cell)
            mesh1 = TriangleMesh(node1, cell)
            mesh0.to_vtk(fname='000.vtu')
            mesh1.to_vtk(fname='111.vtu')


    #print(np.max(np.abs(bf[:-1, :-1] - ff(meshb.entity('node')[:-1, :-1]))))




box = [0, 1, 0, 1]
N = int(sys.argv[1])
h = [1/N, 1/N]
origin = [0, 0]
extend = [0, N+1, 0, N+1]

meshb = UniformMesh2d(extend, h, origin) # 背景网格
mesht = MeshFactory.triangle([0, 1, 0, 1], 0.1)
tnode = mesht.entity('node')
tval = ff(tnode)

#t2b(mesht, meshb, tval)

t0 = time.time()
f = meshb.interpolation_with_sample_points(tnode, tval)
t1 = time.time()
print(t1-t0)

pnode = meshb.entity('node').reshape(-1, 2)

node0 = np.c_[pnode, f.f.reshape(-1)[:, None]]
node1 = np.c_[pnode, ff(pnode)[:, None]]
cell0 = meshb.entity('cell')[:, [0, 2, 1]]
cell1 = meshb.entity('cell')[:, [1, 2, 3]]
cell = np.r_[cell0, cell1]

mesh0 = TriangleMesh(node0, cell)
mesh1 = TriangleMesh(node1, cell)
mesh0.to_vtk(fname='000.vtu')
mesh1.to_vtk(fname='111.vtu')

print(np.max(np.abs(f.f[:-1, :-1] - ff(meshb.entity('node')[:-1, :-1]))))





#fig = plt.figure()
#axes = fig.gca()
#mesht.add_plot(axes)

#fig = plt.figure()
#axes = fig.gca()
#meshb.add_plot(axes)
#plt.show()


