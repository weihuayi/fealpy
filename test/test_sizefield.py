import numpy as np
import sys
import matplotlib.pyplot as plt

from fealpy.mesh import UniformMesh2d, MeshFactory

def ff(x):
    y = np.linalg.norm(x - np.array([[-0.1, -0.1]]), axis=-1)
    #y[:] = 1
    return (2*y)**2


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
    bf[flag] = bf[flag]/numf[flag]
    pnode = meshb.entity('node')
    print(np.max(np.abs(bf[1:-1, 1:-1]-ff(pnode[1:-1, 1:-1]))))


box = [0, 1, 0, 1]
N = int(sys.argv[1])
h = [1/N, 1/N]
origin = [-1/4/N, -1/4/N]
origin = [0, 0]
extend = [0, N+1, 0, N+1]

meshb = UniformMesh2d(extend, h, origin) # 背景网格
mesht = MeshFactory.triangle([0, 1, 0, 1], 0.1)
tnode = mesht.entity('node')
tval = ff(tnode)

t2b(mesht, meshb, tval)
meshb.stiff_matrix()



fig = plt.figure()
axes = fig.gca()
mesht.add_plot(axes)

fig = plt.figure()
axes = fig.gca()
meshb.add_plot(axes)
#plt.show()

