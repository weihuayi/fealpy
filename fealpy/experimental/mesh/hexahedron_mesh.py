from ..backend import backend_manager as bm
from .mesh_base import TensorMesh

class HexahedronMesh(TensorMesh):
    def __init__(self, node, cell):
        super(HexahedronMesh, self).__init__(TD=3)
        self.node = node
        self.cell = cell

        self.localEdge = bm.array([
            (0, 1), (1, 2), (2, 3), (0, 3),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 5), (5, 6), (6, 7), (4, 7)])
        self.localFace = bm.array([
            (0, 3, 2, 1), (4, 5, 6, 7), # bottom and top faces
            (0, 4, 7, 3), (1, 2, 6, 5), # left and right faces
            (0, 1, 5, 4), (2, 3, 7, 6)])# front and back faces
        self.localFace2edge = bm.array([
            (3,  2, 1, 0), (8, 9, 10, 11),
            (4, 11, 7, 3), (1, 6,  9,  5),
            (0,  5, 8, 4), (2, 7, 10,  6)])
        self.localEdge2face = bm.array([
            [4, 0], [3, 0], [5, 0], [0, 2],
            [2, 4], [4, 3], [3, 5], [5, 2],
            [1, 4], [1, 3], [1, 5], [2, 1]])

        self.construct()
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {} 
        self.celldata = {}
        self.meshdata = {}

    def ref_cell_measure(self):
        return 1.0

    def ref_face_meausre(self):
        return 1.0

    def quadrature_formula(self, q, etype='cell'):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature
        qf = GaussLegendreQuadrature(q)
        if etype in {'cell', 3}:
            return TensorProductQuadrature((qf, qf, qf))
        elif etype in {'face', 2}:
            return TensorProductQuadrature((qf, qf))
        elif etype in {'edge', 1}:
            return qf
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def entity_measure(self, etype=3, index=None):
        if etype in {'cell', 3}:
            return self.cell_volume(index=index)
        elif etype in {'face', 2}:
            return self.face_area(index=index)
        elif etype in {'edge', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return bm.zeros(1, dtype=bm.ftype)
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    #def cell_volume(self, index=np.s_[:]):
    #    """
    #    @brief 计算单元的体积, 体积的计算公式为
    #                    \int_c dx = \int_tau |J| d\xi
    #           其中 c 为单元，tau 为参考单元，J 为雅克比矩阵
    #    """
    #    qf = self.integrator(2, etype='cell')
    #    bcs, ws = qf.get_quadrature_points_and_weights()
    #    J = self.jacobi_matrix(bcs, index=index)
    #    detJ = np.linalg.det(J)
    #    val = np.einsum('q, qc->c', ws, detJ)
    #    return val

    #def face_area(self, index=np.s_[:]):
    #    """
    #    @brief 计算面的面积, 面积的计算公式为
    #                    \int_f ds = \int_tau |J| d\xi
    #           其中 f 为面，tau 为参考面，J 为雅克比矩阵
    #    """
    #    qf = self.integrator(2, etype='face')
    #    bcs, ws = qf.get_quadrature_points_and_weights()
    #    J = self.jacobi_matrix(bcs, index=index)
    #    n = np.cross(J[..., 0], J[..., 1], axis=-1)
    #    n = np.sqrt(np.sum(n**2, axis=-1))
    #    val = np.einsum('q, qi->i', ws, n)
    #    return val

    def edge_length(self, index=None):
        """
        @brief 计算边的长度
        """
        edge = self.entity('edge', index=index)
        node = self.entity('node')
        return bm.edge_length(edge, node)

    def bc_to_point(self, bc, index=None):
        """
        @brief 把积分点变换到实际网格实体上的笛卡尔坐标点
        """
        node = self.entity('node')
        if isinstance(bc, tuple) and len(bc) == 3:
            cell = self.entity('cell', index)

            bc0 = bc[0].reshape(-1, 2) # (NQ0, 2)
            bc1 = bc[1].reshape(-1, 2) # (NQ1, 2)
            bc2 = bc[2].reshape(-1, 2) # (NQ2, 2)
            bc = bm.einsum('im, jn, ko->ijkmno', bc0, bc1, bc2).reshape(-1, 8) # (NQ0, NQ1, 2, 2, 2)

            # node[cell].shape == (NC, 8, 3)
            # bc.shape == (NQ, 8)
            p = bm.einsum('...j, cjk->...ck', bc, node[cell[:, [0, 4, 3, 7, 1, 5, 2, 6]]]) # (NQ, NC, 3)

        elif isinstance(bc, tuple) and len(bc) == 2:
            face = self.entity('face', index=index)

            bc0 = bc[0].reshape(-1, 2) # (NQ0, 2)
            bc1 = bc[1].reshape(-1, 2) # (NQ1, 2)
            bc = bm.einsum('im, jn->ijmn', bc0, bc1).reshape(-1, 4) # (NQ0, NQ1, 2, 2)

            # node[cell].shape == (NC, 4, 2)
            # bc.shape == (NQ, 4)
            p = bm.einsum('...j, cjk->...ck', bc, node[face[:, [0, 3, 1, 2]]]) # (NQ, NC, 2)
        else:
            edge = self.entity('edge', index=index)[index]
            p = bm.einsum('...j, ejk->...ek', bc, node[edge]) # (NQ, NE, 2)
        return p
    edge_bc_to_point = bc_to_point
    face_bc_to_point = bc_to_point
    cell_bc_to_point = bc_to_point
   
    def jacobi_matrix(self, bc, index=None):
        """
        @brief 计算参考实体到实际实体间映射的 Jacobi 矩阵。
            x(u, v, w) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        assert isinstance(bc, tuple)
        TD = len(bc)
        node = self.entity('node')
        entity = self.entity(TD, index=index)
        gphi = self.grad_shape_function(bc, p=1, variable='u')
        if TD == 3:
            J = bm.einsum( 'cim, qin->qcmn', node[entity[:, [0, 4, 3, 7, 1, 5, 2, 6]]], gphi)
        elif TD == 2:
            J = bm.einsum( 'cim, qin->qcmn', node[entity[:, [0, 3, 1, 2]]], gphi)
        return J

    def first_fundamental_form(self, J):
        """
        @brief 由 Jacobi 矩阵计算第一基本形式。
        """
        TD = J.shape[-1]
        shape = J.shape[0:-2] + (TD, TD)
        data = [[0 for i in range(TD)] for j in range(TD)]

        for i in range(TD):
            data[i][i] = np.einsum('...d, ...d->...', J[..., i], J[..., i])
            for j in range(i+1, TD):
                data[i][j] = np.einsum('...d, ...d->...', J[..., i], J[..., j])
                data[j][i] = data[i][j]
        data = [val.reshape(val.shape+(1,)) for data_ in data for val in data_]  
        G = np.concatenate(data, axis=-1).reshape(shape)
        return G

    def face_to_ipoint(self, p, index=None):
        """
        @brief 生成每个面上的插值点全局编号
        """
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        edge = self.entity('edge')
        face = self.entity('face')
        face2edge = self.ds.face_to_edge()
        edge2ipoint = self.edge_to_ipoint(p)

        multiIndex = np.zeros([(p+1)**2, 2], dtype=np.int_)
        multiIndex[:, 0] = np.repeat(np.arange(p+1), p+1)
        multiIndex[:, 1] = np.tile(np.arange(p+1), p+1)

        dofidx = np.zeros((4, p+1), dtype=np.int_) #四条边上自由度的局部编号
        dofidx[0], = np.where(multiIndex[:, 1]==0)
        dofidx[1], = np.where(multiIndex[:, 0]==p)
        dofidx[2], = np.where(multiIndex[:, 1]==p)
        dofidx[3], = np.where(multiIndex[:, 0]==0)

        face2ipoint = np.zeros([NF, (p+1)**2], dtype=np.int_)
        localEdge = np.array([[0, 1], [1, 2], [3, 2], [0, 3]], dtype=np.int_)
        for i in range(4): #边上的自由度
            ge = face2edge[:, i]
            idx = np.where(face[:, localEdge[i, 0]] != edge[ge, 0])[0]

            face2ipoint[:, dofidx[i]] = edge2ipoint[ge]
            face2ipoint[idx[:, None], dofidx[i]] = edge2ipoint[ge[idx], ::-1]

        indof = np.all(multiIndex>0, axis=-1)&np.all(multiIndex<p, axis=-1)
        face2ipoint[:, indof] = np.arange(NN+NE*(p-1),
                NN+NE*(p-1)+NF*(p-1)**2).reshape(NF, -1)
        return face2ipoint


    @classmethod
    def from_one_hexahedron(cls):
        """
        @brief 构造一个只有一个六面体的网格
        """
        print('aaa')
        node = bm.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            ], dtype=bm.float64)

        cell = bm.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=bm.int_)
        return cls(node, cell)






