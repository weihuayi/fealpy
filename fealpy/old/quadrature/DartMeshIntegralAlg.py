
from .TetrahedronQuadrature import TetrahedronMesh
from .TriangleQuadrature import TriangleQuadrature 
from .GaussLegendreQuadrature import GaussLegendreQuadrature

class DartMeshIntegralAlg():
    '''!
    @brief DartMesh 上的函数的积分
    '''
    def __init__(self, mesh, q):
        self.q = q
        self.mesh = mesh

        self.edgeintegrator = GaussLegendreQuadrature(q)
        self.faceintegrator = TriangleQuadrature(q)
        self.cellintegrator = TetrahedronQuadrature(q)

        self.edgemeasure = mesh.entity_measure('edge')

    def edge_integral(self, f, index=np.s_[:]):
        '''!
        @brief 对函数 f 在索引为 index 的边上积分
        @param f: {(NQ, NE, 3), index} -> (NQ, NE, ...) 是笛卡尔坐标的函数
        '''

        mesh = self.mesh

        qf = self.edgeintegrator
        bcs, ws = qf.quadpts, qf.weights

        point = mesh.bc_to_point(bcs, index=index) #(NQ, NE, 3)
        fval = f(point, index) # (NQ, NE, ...) 

        em = self.edgemeasure
        val = np.einsum('qe..., e, q->e...', fval, em, ws)
        return val

    def triangle(self, p, bcs):
        '''!
        @brief 计算 p[0], p[1], p[2] 组成的三角形的面积，
            以及每个三角形上 bcs 对应的笛卡尔坐标
        '''
        tm = np.linalg.norm(np.cross(p[1]-p[0], p[2]-p[0]), axis=1)
        point = np.einsum('qi, ifj->qfj', bcs, p)
        return tm, point

    def tetrahedron(self, p, bcs):
        '''!
        @brief 计算 p[0], p[1], p[2], p[3] 组成的四面体的体积，
            以及每个四面体上 bcs 对应的笛卡尔坐标
        '''
        tm = np.sum(np.cross(p[1]-p[0], p[2]-p[0])*(p[3]-p[0]), axis=-1)
        point = np.einsum('qi, ifj->qfj', bcs, p)
        return tm, point

    def face_integral(self, f, index=np.s_[:]):
        '''!
        @brief 对函数 f 在索引为 index 的边上积分
        @param f: {(NQ, NF, 3), index} -> (NQ, NF, ...) 是笛卡尔坐标的函数
        '''
        qf = self.faceintegrator
        bcs, ws = qf.quadpts, qf.weights # bcs: (NQ, 3)

        mesh = self.mesh
        node = mesh.node
        dart = mesh.ds.dart
        hface = mesh.ds.hface[index].copy()

        NF = mesh.number_of_faces()
        index = np.arange(NF)[index] # 拿到整数索引
        NF = len(index)

        shape = f(np.array([[[0, 0, 0]]]), index = np.array([0])).shape
        val = np.zeros(shape[1:], dtype=np.float_)

        isNotOK = np.ones(NF, dtype=np.bool_)
        p = np.zeros([3, NF, 3], dtype=np.float_)
        p[2] = mesh.entity_barycenter('face', index=index) # (NF, 3)
        while np.any(isNotOK):
            p[0, isNotOK] = node[dart[dart[hface[isNotOK], 5], 0]] # (NF, 3)
            p[1, isNotOK] = node[dart[hface[isNotOK], 0]] # (NF, 3)

            tm, point = self.triangle(p[:, isNotOK], bcs)
            fval = f(point, index = index[isNotOK])

            val[isNotOK] = val[isNotOK]+np.einsum('qf..., q, f->f...', fval, ws, tm)

            hface[isNotOK] = dart[hface[isNotOK], 4]
            isNotOK = (hface!=self.ds.hface)
        return val

    def face_integral(self, f, index=np.s_[:]):
        '''!
        @brief 对函数 f 在索引为 index 的边上积分
        @param f: {(NQ, NF, 3), index} -> (NQ, NF, ...) 是笛卡尔坐标的函数
        '''
        qf = self.faceintegrator
        bcs, ws = qf.quadpts, qf.weights # bcs: (NQ, 3)

        mesh = self.mesh
        node = mesh.node
        dart = mesh.ds.dart
        hface = mesh.ds.hface[index]

        NF = mesh.number_of_faces()
        index = np.arange(NF)[index] # 拿到整数索引
        NF = len(index)

        shape = (NF, ) + f(np.array([[[0, 0, 0]]]), index = np.array([0])).shape[2:]
        val = np.zeros(shape, dtype=np.float_)

        isNotOK = np.ones(NF, dtype=np.bool_)
        p = np.zeros([3, NF, 3], dtype=np.float_)
        p[2] = mesh.entity_barycenter('face', index=index) # (NF, 3)
        while np.any(isNotOK):
            p[0, isNotOK] = node[dart[dart[hface[isNotOK], 5], 0]] # (NF, 3)
            p[1, isNotOK] = node[dart[hface[isNotOK], 0]] # (NF, 3)

            tm, point = self.triangle(p[:, isNotOK], bcs)
            fval = f(point, index = index[isNotOK])

            val[isNotOK] = val[isNotOK]+np.einsum('qf..., q, f->f...', fval, ws, tm)

            hface[isNotOK] = dart[hface[isNotOK], 4]
            isNotOK = (hface!=self.ds.hface[index])
        return val

    def cell_integral(self, f, index=np.s_[:]):
        '''!
        @brief 对函数 f 在索引为 index 的边上积分
        @param f: {(NQ, NF, 3), index} -> (NQ, NF, ...) 是笛卡尔坐标的函数
        '''
        qf = self.cellintegrator
        bcs, ws = qf.quadpts, qf.weights # bcs: (NQ, 3)

        mesh = self.mesh
        node = mesh.node
        dart = mesh.ds.dart
        hface = self.hface
        c2f, c2fLoc = mesh.ds.cell_to_face()

        NC = mesh.number_of_cells()
        index = np.arange(NC)[index] # 拿到整数索引
        NC = len(index)

        shape = (NC, ) + f(np.array([[[0, 0, 0]]]), index=np.array([0])).shape[2:]
        val = np.zeros(shape, dtype=np.float_)

        isNotOKCell = np.ones(NC, dtype=np.bool_)
        p = np.zeros([4, NC, 4], dtype=np.float_)
        p[3] = mesh.entity_barycenter('cell', index=index) # (NF, 3)
        start = c2fLoc[:-1].copy()
        while np.any(isNotOKCell):
            cidx = np.where(isNotOKcell)[0]
            face = c2f[start[index[cidx]]]
            p[2, isNotOKCell] = mesh.entity_barycenter('face', index=face)

            isNotOKFace = np.ones(len(face), dtype=np.bool_)
            d = hface[face]
            while np.any(isNotOKFace):
                p[0, cidx[isNotOKFace]] = node[dart[dart[d[isNotOKFace], 5], 0]] # (NF, 3)
                p[1, cidx[isNotOKFace]] = node[dart[d[isNotOKFace], 0]] # (NF, 3)

                tm, point = self.tetrahedron(p[:, isNotOKFace], bcs)
                fval = f(point, index = index[isNotOK])
                val[cidx[isNotOKFace]] = val[cidx[isNotOKFace]]+np.einsum(
                        'qf..., q, f->f...', fval, ws, tm)

                d[isNotOKFace] = dart[d[isNotOKFace], 4]
                isNotOKFace = (d!=hface[face])

            start[index[cidx]] = start[index[cidx]]+1
            isNotOKCell = start<c2fLoc[1:]
        return val


