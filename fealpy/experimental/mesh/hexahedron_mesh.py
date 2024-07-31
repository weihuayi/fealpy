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
        gphi = self.grad_shape_function(bc, p=1, variables='u')
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
            data[i][i] = bm.einsum('...d, ...d->...', J[..., i], J[..., i])
            for j in range(i+1, TD):
                data[i][j] = bm.einsum('...d, ...d->...', J[..., i], J[..., j])
                data[j][i] = data[i][j]
        data = [val.reshape(val.shape+(1,)) for data_ in data for val in data_]  
        G = bm.concatenate(data, axis=-1).reshape(shape)
        return G

    def face_to_ipoint(self, p, index=None):
        """
        @brief 生成每个面上的插值点全局编号
        """
        return self.quad_to_ipoint(p, index) 



    @classmethod
    def from_one_hexahedron(cls):
        """
        @brief 构造一个只有一个六面体的网格
        """
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






