import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from scipy.sparse import triu, tril, find, hstack

from .mesh_base import Mesh, Plotable
from .mesh_data_structure import Mesh3dDataStructure

class HexahedronMeshDataStructure(Mesh3dDataStructure):
    # The following local data structure should be class properties
    localEdge = np.array([
        (0, 1), (1, 2), (2, 3), (0, 3),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 5), (5, 6), (6, 7), (4, 7)])
    localFace = np.array([
        (0, 3, 2, 1), (4, 5, 6, 7), # bottom and top faces
        (0, 4, 7, 3), (1, 2, 6, 5), # left and right faces
        (0, 1, 5, 4), (2, 3, 7, 6)])# front and back faces
    localFace2edge = np.array([
        (3,  2, 1, 0), (8, 9, 10, 11),
        (4, 11, 7, 3), (1, 6,  9,  5),
        (0,  5, 8, 4), (2, 7, 10,  6)])
    localEdge2face = np.array([
        [4, 0], [3, 0], [5, 0], [0, 2],
        [2, 4], [4, 3], [3, 5], [5, 2],
        [1, 4], [1, 3], [1, 5], [2, 1]])
    ccw = np.array([0, 1, 2, 3])


class HexahedronMesh(Mesh, Plotable):
    """
    @brief 非结构六面体网格数据结构对象
    """
    def __init__(self, node, cell):
        self.node = node
        NN = node.shape[0]
        self.ds = HexahedronMeshDataStructure(NN, cell)

        self.meshtype = 'hex'
        self.meshtype = 'HEX'

        self.itype = cell.dtype
        self.ftype = node.dtype

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {}
        self.meshdata = {}

        self.edge_shape_function = self._shape_function
        self.grad_edge_shape_function = self._grad_shape_function

    def ref_cell_measure(self):
        return 1.0

    def ref_face_meausre(self):
        return 1.0

    def integrator(self, q, etype='cell'):
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

    def entity_measure(self, etype=3, index=np.s_[:]):
        if etype in {'cell', 3}:
            return self.cell_volume(index=index)
        elif etype in {'face', 2}:
            return self.face_area(index=index)
        elif etype in {'edge', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return np.zeros(1, dtype=self.ftype)
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def cell_volume(self, index=np.s_[:]):
        """
        @brief 计算单元体积
        """
        qf = self.integrator(2, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = self.jacobi_matrix(bcs, index=index)
        detJ = np.linalg.det(J)
        val = np.einsum('q, qc->c', ws, detJ)
        return val

    def face_area(self, index=np.s_[:]):
        """
        @brief
        """
        qf = self.integrator(2, etype='face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = self.jacobi_matrix(bcs, index=index)
        n = np.cross(J[..., 0], J[..., 1], axis=-1)
        n = np.sqrt(np.sum(n**2, axis=-1))
        val = np.einsum('q, qi->i', ws, n)
        return val

    def bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把积分点变换到实际网格实体上的笛卡尔坐标点
        """
        node = self.entity('node')
        if isinstance(bc, tuple) and len(bc) == 3:
            cell = self.entity('cell')[index]

            bc0 = bc[0].reshape(-1, 2) # (NQ0, 2)
            bc1 = bc[1].reshape(-1, 2) # (NQ1, 2)
            bc2 = bc[2].reshape(-1, 2) # (NQ2, 2)
            bc = np.einsum('im, jn, ko->ijkmno', bc0, bc1, bc2).reshape(-1, 8) # (NQ0, NQ1, 2, 2, 2)

            # node[cell].shape == (NC, 8, 3)
            # bc.shape == (NQ, 8)
            p = np.einsum('...j, cjk->...ck', bc, node[cell[:, [0, 4, 3, 7, 1, 5, 2, 6]]]) # (NQ, NC, 3)

        elif isinstance(bc, tuple) and len(bc) == 2:
            face = self.entity('face', index=index)

            bc0 = bc[0].reshape(-1, 2) # (NQ0, 2)
            bc1 = bc[1].reshape(-1, 2) # (NQ1, 2)
            bc = np.einsum('im, jn->ijmn', bc0, bc1).reshape(-1, 4) # (NQ0, NQ1, 2, 2)

            # node[cell].shape == (NC, 4, 2)
            # bc.shape == (NQ, 4)
            p = np.einsum('...j, cjk->...ck', bc, node[face[:, [0, 3, 1, 2]]]) # (NQ, NC, 2)
        else:
            edge = self.entity('edge', index=index)[index]
            p = np.einsum('...j, ejk->...ek', bc, node[edge]) # (NQ, NE, 2)
        return p

    edge_bc_to_point = bc_to_point
    face_bc_to_point = bc_to_point
    cell_bc_to_point = bc_to_point

    def shape_function(self, bc, p=1):
        """
        @brief 六面体单元上的形函数
        """
        assert isinstance(bc, tuple)
        TD = len(bc)
        phi = [self._shape_function(val, p=p) for val in bc]
        ldof = (p+1)**TD
        return np.einsum('im, jn, ko->ijkmno', phi[0], phi[1], phi[2]).reshape(-1, ldof)

    cell_shape_function = shape_function

    def face_shape_function(self, bc, p=1):
        """
        @brief 四边形面上的形函数
        """
        TD = len(bc)
        phi = [self._shape_function(val, p=p) for val in bc]
        ldof = (p+1)**TD
        return np.einsum('im, jn->ijmn', phi[0], phi[1]).reshape(-1, ldof)

    def grad_shape_function(self, bc, p=1, variables='x', index=np.s_[:]):
        """
        @brief  六面体单元形函数的导数
        """
        assert isinstance(bc, tuple)
        TD = len(bc)
        Dlambda = np.array([-1, 1], dtype=self.ftype)
        phi = self._shape_function(bc[0], p=p)
        R = self._grad_shape_function(bc[0], p=p)
        dphi = np.einsum('...ij, j->...i', R, Dlambda) # (..., ldof)

        n = phi.shape[0]**TD
        ldof = phi.shape[-1]**TD
        shape = (n, ldof, TD)
        gphi = np.zeros(shape, dtype=self.ftype)

        if TD == 3:
            gphi[..., 0] = np.einsum('im, jn, ko->ijkmno', dphi, phi, phi).reshape(-1, ldof)
            gphi[..., 1] = np.einsum('im, jn, ko->ijkmno', phi, dphi, phi).reshape(-1, ldof)
            gphi[..., 2] = np.einsum('im, jn, ko->ijkmno', phi, phi, dphi).reshape(-1, ldof)
            if variables == 'x':
                J = self.jacobi_matrix(bc, index=index)
                J = np.linalg.inv(J)
                # J^{-T}\nabla_u phi
                gphi = np.einsum('qcmn, qlm->qcln', J, gphi)
                return gphi
        elif TD == 2:
            gphi[..., 0] = np.einsum('im, jn->ijmn', dphi, phi).reshape(-1, ldof)
            gphi[..., 1] = np.einsum('im, jn->ijmn', phi, dphi).reshape(-1, ldof)
            if variables == 'x':
                J = self.jacobi_matrix(bc, index=index)
                G = self.first_fundamental_form(J)
                G = np.linalg.inv(G)
                gphi = np.einsum('qikm, qimn, qln->qilk', J, G, gphi)
                return gphi
        return gphi

    cell_grad_shape_function = grad_shape_function

    def jacobi_matrix(self, bc, index=np.s_[:]):
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
            J = np.einsum( 'cim, qin->qcmn', node[entity[:, [0, 4, 3, 7, 1, 5, 2, 6]]], gphi)
        elif TD == 2:
            J = np.einsum( 'cim, qin->qcmn', node[entity[:, [0, 3, 1, 2]]], gphi)
        return J

    def first_fundamental_form(self, J):
        """
        @brief 由 Jacobi 矩阵计算第一基本形式。
        """
        TD = J.shape[-1]
        shape = J.shape[0:-2] + (TD, TD)
        G = np.zeros(shape, dtype=self.ftype)
        for i in range(TD):
            G[..., i, i] = np.einsum('...d, ...d->...', J[..., i], J[..., i])
            for j in range(i+1, TD):
                G[..., i, j] = np.einsum('...d, ...d->...', J[..., i], J[..., j])
                G[..., j, i] = G[..., i, j]
        return G

    def uniform_refine(self, n=1):
        """
        @brief 一致加密六面体网格 n 次
        """
        for i in range(n):
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NF = self.number_of_faces()
            NC = self.number_of_cells()
            node = np.zeros((NN + NE + NF + NC, 3), dtype=self.ftype)
            start = 0
            end = NN
            node[start:end] = self.entity('node')
            start = end
            end = start + NE
            node[start:end] = self.entity_barycenter('edge')
            start = end
            end = start + NF
            node[start:end] = self.entity_barycenter('face')
            start = end
            end = start + NF
            node[start:end] = self.entity_barycenter('cell')

            cell = np.zeros((8*NC, 8), dtype=self.itype)
            c2n = self.entity('cell')
            c2e = self.ds.cell_to_edge() + NN
            c2f = self.ds.cell_to_face() + (NN + NE)
            c2c = np.arange(NC) + (NN + NE + NF)

            cell[0::8, 0] = c2n[:, 0]
            cell[0::8, 1] = c2e[:, 0]
            cell[0::8, 2] = c2f[:, 0]
            cell[0::8, 3] = c2e[:, 3]
            cell[0::8, 4] = c2e[:, 4]
            cell[0::8, 5] = c2f[:, 4]
            cell[0::8, 6] = c2c
            cell[0::8, 7] = c2f[:, 2]

            cell[1::8, 0] = c2n[:, 1]
            cell[1::8, 1] = c2e[:, 1]
            cell[1::8, 2] = c2f[:, 0]
            cell[1::8, 3] = c2e[:, 0]
            cell[1::8, 4] = c2e[:, 5]
            cell[1::8, 5] = c2f[:, 3]
            cell[1::8, 6] = c2c
            cell[1::8, 7] = c2f[:, 4]

            cell[2::8, 0] = c2n[:, 2]
            cell[2::8, 1] = c2e[:, 2]
            cell[2::8, 2] = c2f[:, 0]
            cell[2::8, 3] = c2e[:, 1]
            cell[2::8, 4] = c2e[:, 6]
            cell[2::8, 5] = c2f[:, 5]
            cell[2::8, 6] = c2c
            cell[2::8, 7] = c2f[:, 3]

            cell[3::8, 0] = c2n[:, 3]
            cell[3::8, 1] = c2e[:, 3]
            cell[3::8, 2] = c2f[:, 0]
            cell[3::8, 3] = c2e[:, 2]
            cell[3::8, 4] = c2e[:, 7]
            cell[3::8, 5] = c2f[:, 2]
            cell[3::8, 6] = c2c
            cell[3::8, 7] = c2f[:, 5]

            cell[4::8, 0] = c2n[:, 4]
            cell[4::8, 1] = c2e[:,11]
            cell[4::8, 2] = c2f[:, 1]
            cell[4::8, 3] = c2e[:, 8]
            cell[4::8, 4] = c2e[:, 4]
            cell[4::8, 5] = c2f[:, 2]
            cell[4::8, 6] = c2c
            cell[4::8, 7] = c2f[:, 4]

            cell[5::8, 0] = c2n[:, 5]
            cell[5::8, 1] = c2e[:, 8]
            cell[5::8, 2] = c2f[:, 1]
            cell[5::8, 3] = c2e[:, 9]
            cell[5::8, 4] = c2e[:, 5]
            cell[5::8, 5] = c2f[:, 4]
            cell[5::8, 6] = c2c
            cell[5::8, 7] = c2f[:, 3]

            cell[6::8, 0] = c2n[:, 6]
            cell[6::8, 1] = c2e[:, 9]
            cell[6::8, 2] = c2f[:, 1]
            cell[6::8, 3] = c2e[:,10]
            cell[6::8, 4] = c2e[:, 6]
            cell[6::8, 5] = c2f[:, 3]
            cell[6::8, 6] = c2c
            cell[6::8, 7] = c2f[:, 5]

            cell[7::8, 0] = c2n[:, 7]
            cell[7::8, 1] = c2e[:,10]
            cell[7::8, 2] = c2f[:, 1]
            cell[7::8, 3] = c2e[:,11]
            cell[7::8, 4] = c2e[:, 7]
            cell[7::8, 5] = c2f[:, 5]
            cell[7::8, 6] = c2c
            cell[7::8, 7] = c2f[:, 2]

            self.node = node
            self.ds.reinit(NN+NE+NF+NC, cell)


    def number_of_local_ipoints(self, p, iptype='cell'):
        if iptype in {'cell', 3}:
            return (p+1)**3
        elif iptype in {'face', 2}:
            return (p + 1)**2
        elif iptype in {'edge', 1}:
            return p + 1
        elif iptype in {'node', 0}:
            return 1

    def number_of_global_ipoints(self, p):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        NC = self.number_of_cells()
        return NN + NE * (p-1) + NF * (p-1)**2 + NC * (p-1)**3

    def prolongation_matrix(self, p0:int, p1:int):
        """
        @brief 生成从 p0 元到 p1 元的延拓矩阵，假定 0 < p0 < p1

        TODO: 测试程序 
        """

        assert 0 < p0 < p1

        TD = self.top_dimension()

        gdof1 = self.number_of_global_ipoints(p1)
        gdof0 = self.number_of_global_ipoints(p0)

        # 1. 网格节点上的插值点 
        NN = self.number_of_nodes()
        I = range(NN)
        J = range(NN)
        V = np.ones(NN, dtype=self.ftype)
        P = coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 2. 网格边内部的插值点 
        NE = self.number_of_edges()
        # p1 元在边上插值点对应的重心坐标
        bcs = self.multi_index_matrix(p1, 1)/p1 
        # p0 元基函数在 p1 元对应的边内部插值点处的函数值
        phi = self.edge_shape_function(bcs[1:-1], p=p0) # (ldof1 - 2, ldof0)  
       
        e2p1 = self.edge_to_ipoint(p1)[:, 1:-1]
        e2p0 = self.edge_to_ipoint(p0)

        shape = (NE, ) + phi.shape
        I = np.broadcast_to(e2p1[:, :, None], shape=shape).flat
        J = np.broadcast_to(e2p0[:, None, :], shape=shape).flat
        V = np.broadcast_to( phi[None, :, :], shape=shape).flat

        P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 3. 网格面内部的插值点
        NF = self.number_of_faces()
        # p1 元在单元上对应插值点的重心坐标
        bcs = self.multi_index_matrix(p1, 1)/p1
        # p0 元基函数在 p1 元对应的单元内部插值点处的函数值
        phi = self.face_shape_function((bcs[1:-1], bcs[1:-1]), p=p0) #
        f2p1 = self.face_to_ipoint(p1).reshape(NF, p1+1, p1+1)[:, 1:-1, 1:-1]
        f2p1 = c2p1.reshape(NF, -1)
        f2p0 = self.face_to_ipoint(p0)

        shape = (NF, ) + phi.shape
        I = np.broadcast_to(f2p1[:, :, None], shape=shape).flat
        J = np.broadcast_to(f2p0[:, None, :], shape=shape).flat
        V = np.broadcast_to( phi[None, :, :], shape=shape).flat

        P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 4. 网格单元内部的插值点
        NC = self.number_of_cells()
        # p1 元在单元上对应插值点的重心坐标
        bcs = self.multi_index_matrix(p1, 1)/p1
        # p0 元基函数在 p1 元对应的单元内部插值点处的函数值
        phi = self.cell_shape_function((bcs[1:-1], bcs[1:-1], bcs[1:-1]), p=p0) #
        c2p1 = self.cell_to_ipoint(p1).reshape(NF, p1+1, p1+1, p1+1)[:, 1:-1, 1:-1, 1:-1]
        c2p1 = c2p1.reshape(NC, -1)
        c2p0 = self.cell_to_ipoint(p0)

        shape = (NC, ) + phi.shape

        I = np.broadcast_to(c2p1[:, :, None], shape=shape).flat
        J = np.broadcast_to(c2p0[:, None, :], shape=shape).flat
        V = np.broadcast_to( phi[None, :, :], shape=shape).flat

        P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        return P.tocsr()

    def interpolation_points(self, p, index=np.s_[:]):
        """
        @brief 生成整个网格上的插值点
        """
        node = self.entity('node')
        cell = self.entity('cell')
        NC = self.number_of_cells()

        c2ip = self.cell_to_ipoint(p)
        gp = self.number_of_global_ipoints(p)
        ipoint = np.zeros([gp, 3], dtype=np.float64)

        p04 = np.linspace(node[cell[:, 0]], node[cell[:, 4]], p+1, endpoint=True).swapaxes(0, 1)
        p37 = np.linspace(node[cell[:, 3]], node[cell[:, 7]], p+1, endpoint=True).swapaxes(0, 1)
        p15 = np.linspace(node[cell[:, 1]], node[cell[:, 5]], p+1, endpoint=True).swapaxes(0, 1)
        p26 = np.linspace(node[cell[:, 2]], node[cell[:, 6]], p+1, endpoint=True).swapaxes(0, 1)

        p0 = np.linspace(p04, p37, p+1, endpoint=True).swapaxes(0, 1).reshape(NC, -1, 3)
        p1 = np.linspace(p15, p26, p+1, endpoint=True).swapaxes(0, 1).reshape(NC, -1, 3)
        ipoint[c2ip] = np.linspace(p0, p1, p+1, endpoint=True).swapaxes(0, 1).reshape(NC, -1, 3)
        return ipoint

    def face_to_ipoint(self, p, index=np.s_[:]):
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

    def cell_to_ipoint(self, p, index=np.s_[:]):
        """!
        @brief 生成每个单元上的插值点全局编号
        """

        cell = self.entity('cell', index=index)
        if p == 1:
            return cell[:, [0, 4, 3, 7, 1, 5, 2, 6]]

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        NC = self.number_of_cells()

        edge = self.entity('edge')
        face = self.entity('face')

        cell2face = self.ds.cell_to_face()
        face2edge = self.ds.face_to_edge()
        cell2edge = self.ds.cell_to_edge()

        face2ipoint = self.face_to_ipoint(p)

        multiIndex = np.zeros([(p+1)**3, 3], dtype=np.int_)
        multiIndex[:, 0] = np.repeat(np.arange(p+1), (p+1)**2)
        multiIndex[:, 1] = np.tile(np.repeat(np.arange(p+1), p+1), (p+1))
        multiIndex[:, 2] = np.tile(np.arange(p+1), (p+1)**2)

        dofidx = np.zeros((6, (p+1)**2), dtype=np.int_) #四条边上自由度的局部编号
        dofidx[0], = np.where(multiIndex[:, 2]==0)
        dofidx[1], = np.where(multiIndex[:, 2]==p)
        dofidx[2], = np.where(multiIndex[:, 0]==0)
        dofidx[3], = np.where(multiIndex[:, 0]==p)
        dofidx[4], = np.where(multiIndex[:, 1]==0)
        dofidx[5], = np.where(multiIndex[:, 1]==p)

        cell2ipoint = np.zeros([NC, (p+1)**3], dtype=np.int_)
        lf2e = np.array([[0, 1, 2, 3], [8, 9, 10, 11],
                         [3, 7, 11, 4], [1, 6, 9, 5],
                         [0, 5, 8, 4], [2, 6, 10, 7]], dtype=np.int_)

        multiIndex2d = multiIndex[:(p+1)**2, 1:]
        multiIndex2d = np.c_[multiIndex2d, p-multiIndex2d]

        lf2e = lf2e[:, [3, 0, 1, 2]]
        face2edge = face2edge[:, [3, 0, 1, 2]]
        for i in range(6): #面上的自由度
            gfe = face2edge[cell2face[:, i]]
            lfe = cell2edge[:, lf2e[i]]
            idx0 = np.argsort(gfe, axis=-1)
            idx1 = np.argsort(lfe, axis=-1)
            idx1 = np.argsort(idx1, axis=-1)
            idx0 = idx0[np.arange(NC)[:, None], idx1] #(NC, 4)
            idx = multiIndex2d[:, idx0].swapaxes(0, 1) #(NC, NQ, 4)

            idx = idx[..., 0]*(p+1)+idx[..., 1]
            cell2ipoint[:, dofidx[i]] = face2ipoint[cell2face[:, i, None], idx]

        indof = np.all(multiIndex>0, axis=-1)&np.all(multiIndex<p, axis=-1)
        cell2ipoint[:, indof] = np.arange(NN+NE*(p-1)+NF*(p-1)**2,
                NN+NE*(p-1)+NF*(p-1)**2+NC*(p-1)**3).reshape(NC, -1)
        return cell2ipoint[index]

    @classmethod
    def from_one_hexahedron(cls, twist=False):
        """
        @brief 构造一个只有一个六面体的网格
        """
        node = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            ], dtype=np.float64)

        if twist:
            upnode = node[4:]
            upnode -= np.array([[0.5, 0.5, 1]])
            upnode = np.cross(np.array([[0, 0, 1]]), upnode)
            node[4:] = upnode + np.array([[0.5, 0.5, 1]])

        cell = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int_)
        return cls(node, cell)

    @classmethod
    def from_one_tetrahedron(cls):
        """
        @brief 把一个四面体区域分解为四个六面体单元
        """
        from .tetrahedron_mesh import TetrahedronMesh

        mesh = TetrahedronMesh.from_one_tetrahedron(meshtype='equ')
        return cls.from_tetrahedron_mesh(mesh)


    @classmethod
    def from_tetrahedron_mesh(cls, mesh):
        """
        @brief 给定一个四面体网格，把每个四面体网格分成四个六面体
        """
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()
        node = np.zeros((NN + NE + NF + NC, 3), dtype=mesh.ftype)
        start = 0
        end = NN
        node[start:end] = mesh.entity('node')
        start = end
        end = start + NE
        node[start:end] = mesh.entity_barycenter('edge')
        start = end
        end = start + NF
        node[start:end] = mesh.entity_barycenter('face')
        start = end
        end = start + NF
        node[start:end] = mesh.entity_barycenter('cell')

        cell = np.zeros((4*NC, 8), dtype=mesh.itype)
        c2n = mesh.entity('cell')
        c2e = mesh.ds.cell_to_edge() + NN
        c2f = mesh.ds.cell_to_face() + (NN + NE)
        c2c = np.arange(NC) + (NN + NE + NF)

        cell[0::4, 0] = c2n[:, 0]
        cell[0::4, 1] = c2e[:, 0]
        cell[0::4, 2] = c2f[:, 3]
        cell[0::4, 3] = c2e[:, 1]
        cell[0::4, 4] = c2e[:, 2]
        cell[0::4, 5] = c2f[:, 2]
        cell[0::4, 6] = c2c
        cell[0::4, 7] = c2f[:, 1]

        cell[1::4, 0] = c2n[:, 1]
        cell[1::4, 1] = c2e[:, 3]
        cell[1::4, 2] = c2f[:, 3]
        cell[1::4, 3] = c2e[:, 0]
        cell[1::4, 4] = c2e[:, 4]
        cell[1::4, 5] = c2f[:, 0]
        cell[1::4, 6] = c2c
        cell[1::4, 7] = c2f[:, 2]

        cell[2::4, 0] = c2n[:, 2]
        cell[2::4, 1] = c2e[:, 1]
        cell[2::4, 2] = c2f[:, 3]
        cell[2::4, 3] = c2e[:, 3]
        cell[2::4, 4] = c2e[:, 5]
        cell[2::4, 5] = c2f[:, 1]
        cell[2::4, 6] = c2c
        cell[2::4, 7] = c2f[:, 0]

        cell[3::4, 0] = c2n[:, 3]
        cell[3::4, 1] = c2e[:, 5]
        cell[3::4, 2] = c2f[:, 0]
        cell[3::4, 3] = c2e[:, 4]
        cell[3::4, 4] = c2e[:, 2]
        cell[3::4, 5] = c2f[:, 1]
        cell[3::4, 6] = c2c
        cell[3::4, 7] = c2f[:, 2]

        return cls(node, cell)



    @classmethod
    def from_box(cls, box=[0, 1, 0, 1, 0, 1], nx=10, ny=10, nz=10, threshold=None):
        """
        Generate a hexahedral mesh for a box domain.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param nz Number of divisions along the z-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return HexahedronMesh instance
        """
        NN = (nx+1)*(ny+1)*(nz+1)
        NC = nx*ny*nz
        node = np.zeros((NN, 3), dtype=np.float64)
        X, Y, Z = np.mgrid[
                box[0]:box[1]:(nx+1)*1j,
                box[2]:box[3]:(ny+1)*1j,
                box[4]:box[5]:(nz+1)*1j
                ]
        node[:, 0] = X.flat
        node[:, 1] = Y.flat
        node[:, 2] = Z.flat

        idx = np.arange(NN).reshape(nx+1, ny+1, nz+1)
        c = idx[:-1, :-1, :-1]

        cell = np.zeros((NC, 8), dtype=np.int_)
        nyz = (ny + 1)*(nz + 1)
        cell[:, 0] = c.flatten()
        cell[:, 1] = cell[:, 0] + nyz
        cell[:, 2] = cell[:, 1] + nz + 1
        cell[:, 3] = cell[:, 0] + nz + 1
        cell[:, 4] = cell[:, 0] + 1
        cell[:, 5] = cell[:, 4] + nyz
        cell[:, 6] = cell[:, 5] + nz + 1
        cell[:, 7] = cell[:, 4] + nz + 1

        if threshold is not None:
            bc = np.sum(node[cell, :], axis=1)/cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = np.zeros(NN, dtype=np.bool_)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = np.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]

        return cls(node, cell)

    @classmethod
    def from_unit_cube(cls, nx=10, ny=10, nz=10, threshold=None):
        """
        Generate a hexahedral mesh for a unit cube.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param nz Number of divisions along the z-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return HexahedronMesh instance
        """
        return cls.from_box(box=[0, 1, 0, 1, 0, 1], nx=nx, ny=ny, nz=nz, threshold=threshold)
    
    ## @ingroup MeshGenerators
    @classmethod
    def from_fuel_rod_gmsh(cls,R1,R2,L,w,h,l,p):
        """
        Generate a hexahedron mesh for a fuel-rod region by gmsh

        @param R1 The radius of semicircles
        @param R2 The radius of quarter circles
        @param L The length of straight segments
        @param w The thickness of caldding
        @param h Parameter controlling mesh density
        @param l The length of the fuel-rod
        @param p The pitch of the fuel-rod
        @return HexahedronMesh instance
        """
        import gmsh
        import math
        gmsh.initialize()
        gmsh.model.add("fuel_rod_3D")

        # 内部单元大小
        Lc1 = h
        # 包壳单元大小
        Lc2 = h/2.5

        factory = gmsh.model.geo
        # 外圈点
        factory.addPoint( -R1 -R2 -L, 0 , 0 , Lc2 , 1 )#圆心1
        factory.addPoint( -R1 -R2 -L, -R1 , 0 , Lc2 , 2)
        factory.addPoint( -R1 -R2 , -R1 , 0 , Lc2 , 3)
        factory.addPoint( -R1 -R2 , -R1 -R2 , 0 , Lc2 , 4)#圆心2
        factory.addPoint( -R1 , -R1 -R2 , 0 , Lc2 , 5)
        factory.addPoint( -R1 , -R1 -R2 -L , 0 , Lc2 , 6)
        factory.addPoint( 0 , -R1 -R2 -L , 0 , Lc2 , 7)#圆心3
        factory.addPoint( R1 , -R1 -R2 -L , 0 , Lc2 , 8)
        factory.addPoint( R1 , -R1 -R2 , 0 , Lc2 , 9)
        factory.addPoint( R1 +R2 , -R1 -R2 , 0, Lc2 , 10)#圆心4
        factory.addPoint( R1 +R2 , -R1 , 0 , Lc2 , 11) 
        factory.addPoint( R1 +R2 +L , -R1 , 0 , Lc2 , 12)
        factory.addPoint( R1 +R2 +L , 0 , 0 , Lc2 , 13)#圆心5
        factory.addPoint( R1 +R2 +L , R1 , 0 , Lc2 , 14)
        factory.addPoint( R1 +R2 , R1 , 0 , Lc2 , 15)
        factory.addPoint( R1 +R2 , R1 +R2 , 0 , Lc2 , 16)#圆心6
        factory.addPoint( R1 , R1 +R2 , 0 , Lc2 , 17)
        factory.addPoint( R1 , R1 +R2 +L , 0 , Lc2 , 18)
        factory.addPoint( 0 , R1 +R2 +L , 0 , Lc2 , 19)#圆心7
        factory.addPoint( -R1 , R1 +R2 +L , 0 , Lc2 , 20)
        factory.addPoint( -R1 , R1 +R2 , 0 , Lc2 , 21)
        factory.addPoint( -R1 -R2 , R1 +R2 , 0 , Lc2 , 22)#圆心8
        factory.addPoint( -R1 -R2 , R1 , 0 , Lc2 , 23)
        factory.addPoint( -R1 -R2 -L , R1 , 0 , Lc2 , 24)

        # 外圈线
        line_list_out = []
        for i in range(8):
            if i == 0:
                factory.addCircleArc(24 , 3*i+1 , 3*i+2, 2*i+1)
                factory.addLine( 3*i+2 , 3*i+3 , 2*(i+1) )
            else:
                factory.addCircleArc(3*i , 3*i+1 , 3*i+2 , 2*i+1)
                factory.addLine( 3*i+2 , 3*i+3 , 2*(i+1) )
            # 填充线环中的线
            line_list_out.append(2*i+1)
            line_list_out.append(2*(i+1))
        # 生成外圈线环
        factory.addCurveLoop(line_list_out,17)

        # 内圈点
        factory.addPoint( -R1 -R2 -L, -R1 +w , 0 , Lc1 , 25)
        factory.addPoint( -R1 -R2 , -R1 +w , 0 , Lc1 , 26)
        factory.addPoint( -R1 +w , -R1 -R2 , 0 , Lc1 , 27)
        factory.addPoint( -R1 +w , -R1 -R2 -L , 0 , Lc1 , 28)
        factory.addPoint( R1 -w , -R1 -R2 -L , 0 , Lc1 , 29)
        factory.addPoint( R1 -w , -R1 -R2 , 0 , Lc1 , 30)
        factory.addPoint( R1 +R2 , -R1 +w , 0 , Lc1 , 31) 
        factory.addPoint( R1 +R2 +L , -R1 +w , 0 , Lc1 , 32)
        factory.addPoint( R1 +R2 +L , R1 -w , 0 , Lc1 , 33)
        factory.addPoint( R1 +R2 , R1 -w , 0 , Lc1 , 34)
        factory.addPoint( R1 -w , R1 +R2 , 0 , Lc1 , 35)
        factory.addPoint( R1 -w , R1 +R2 +L , 0 , Lc1 , 36)
        factory.addPoint( -R1 +w , R1 +R2 +L , 0 , Lc1 , 37)
        factory.addPoint( -R1 +w , R1 +R2 , 0 , Lc1 , 38)
        factory.addPoint( -R1 -R2 , R1 -w, 0 , Lc1 , 39)
        factory.addPoint( -R1 -R2 -L , R1 -w, 0 , Lc1 , 40)

        # 内圈线
        line_list_in = []
        for j in range(8):
            if j == 0:
                factory.addCircleArc(40 , 3*j+1 , 25+2*j , 18+2*j)
                factory.addLine(25+2*j , 26+2*j , 19+2*j)
            else:
                factory.addCircleArc(24+2*j , 3*j+1 , 25+2*j, 18+2*j)
                factory.addLine(25+2*j , 26+2*j , 19+2*j)
            line_list_in.append(18+2*j)
            line_list_in.append(19+2*j)
        # 生成内圈线环  
        factory.addCurveLoop(line_list_in,34)

        # 内圈面
        factory.addPlaneSurface([34],35)
        # 包壳截面
        factory.addPlaneSurface([17, 34],36)

        factory.synchronize()

        N = math.ceil((2*l)/p)
        angle = ((2*l)/p* math.pi) / N
        nsection = math.ceil(l/(N*h))
        for i in range(N):
            if i == 0:
                ov1 = factory.twist([(2,35)],0,0,0,0,0,l/N,0,0,1,angle,[nsection],[],True)
                ov2 = factory.twist([(2,36)],0,0,0,0,0,l/N,0,0,1,angle,[nsection],[],True)
            else:
                ov1 = factory.twist([(2,ov1[0][1])],0,0,0,0,0,l/N,0,0,1,angle,[nsection],[],True)
                ov2 = factory.twist([(2,ov2[0][1])],0,0,0,0,0,l/N,0,0,1,angle,[nsection],[],True)

        factory.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        # 生成网格
        gmsh.model.mesh.generate(3)
        #gmsh.fltk.run()
        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = np.array(node_coords, dtype=np.float64).reshape(-1, 3)

        #节点的编号映射
        nodetags_map = dict({j:i for i,j in enumerate(node_tags)})

        # 获取四面体单元信息
        Hexahedron_type = 5  
        Hexahedron_tags, Hexahedron_connectivity = gmsh.model.mesh.getElementsByType(Hexahedron_type)
        evid = np.array([nodetags_map[j] for j in Hexahedron_connectivity])
        cell = evid.reshape((Hexahedron_tags.shape[-1],-1))

        gmsh.finalize()
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of cells: {cell.shape[0]}")

        return cls(node,cell)

HexahedronMesh.set_ploter('3d')
