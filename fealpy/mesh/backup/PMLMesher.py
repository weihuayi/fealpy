import numpy as np

def SpherePMLMesher(mesh, fidx=None, l=5, h=1):
    '''!
    @brief 生成 mesh 在 fidx 对应面延伸出去的 PML 网格
    @param fidx PML 面索引, 默认是所有边界面 
    @param l PML 层数
    @param h PML 厚度
    '''
    if fidx is None:
        fidx = mesh.boundary_face_index()

    node = mesh.entity('node')
    face = mesh.entity('face')[fidx]
    NF = len(face) #表面面的个数

    nidx = np.unique(face)
    NN = len(nidx) #表面点的个数
    nidxmap = np.zeros(len(node), dtype=np.int_)
    nidxmap[nidx] = np.arange(NN)

    tface = nidxmap[face]
    tnode = node[nidx] 

    vn = np.zeros([NN, 3], dtype=np.float_)
    fn = mesh.face_unit_normal(index=fidx)
    fn = np.tile(fn, 3).reshape(NF, 3, 3)

    np.add.at(vn, tface, fn)
    vn = vn/np.linalg.norm(vn, axis=-1)[:, None]

    pcell = np.zeros([NF*l, 6], dtype = face.dtype)
    pnode = np.zeros([NN*(l+1), 3], dtype = node.dtype)
    for i in range(l):
        pcell[i*NF:(i+1)*NF, :3] = tface + i*NN
        pcell[i*NF:(i+1)*NF, 3:] = tface + (i+1)*NN
        pnode[i*NN:(i+1)*NN] = tnode + i*vn*h/l
    pnode[-NN:] = tnode + vn*h

    pnidxmap = np.zeros(NN*(l+1), dtype=np.int_)
    pnidxmap[tface] = face
    pnidxmap[NN:] = pnidxmap[NN:] + len(node)

    cell = pnidxmap[pcell]
    node = np.r_[node, pnode]
    return node, cell

