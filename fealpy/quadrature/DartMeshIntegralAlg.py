
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
        tm = np.linalg.norm(np.cross(p[0]-p[2], p[1]-p[2]), axis=1)
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

        val = np.zeros([NF,])

        isNotOK = np.ones(NF, dtype=np.bool_)
        p = np.zeros([3, NF, 3], dtype=np.float_)
        p[2] = mesh.entity_barycenter('face', index=index) # (NF, 3)
        while np.any(isNotOK):
            p[0, isNotOK] = node[dart[dart[hface[isNotOK], 5], 0]] # (NF, 3)
            p[1, isNotOK] = node[dart[hface[isNotOK], 0]] # (NF, 3)

            tm, point = self.triangle(p[:, isNotOK], bcs)
            fval = f(point, index = index[isNotOK])

            hface[isNotOK] = dart[hface[isNotOK], 4]
            isNotOK = (hface!=self.ds.hface)




        point = mesh.bc_to_point(bcs, index=index) #(NQ, NE, 3)
        fval = f(point, index) # (NQ, NE, ...) 

        em = self.edgemeasure
        val = np.einsum('qe..., e, q->e...', fval, em, ws)
























