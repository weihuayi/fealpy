# from test.mesh.tetrahedron_mesh_data import from_box
from typing import Union, Optional
from math import sqrt
from ..backend import backend_manager as bm
from .mesh_base import TensorMesh
from ..typing import TensorLike, Index, _S
from .plot import Plotable

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

class HexahedronMesh(TensorMesh, Plotable):
    def __init__(self, node, cell):
        super(HexahedronMesh, self).__init__(TD = 3,
                                        itype = cell.dtype, ftype = node.dtype)
        self.node = node
        self.cell = cell

        self.meshtype = 'hex'
        self.p = 1

        kwargs = bm.context(cell)

        self.localEdge = bm.array([
            (0, 1), (1, 2), (2, 3), (0, 3),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 5), (5, 6), (6, 7), (4, 7)], **kwargs)
        self.localFace = bm.array([
            (0, 3, 2, 1), (4, 5, 6, 7), # bottom and top faces
            (0, 4, 7, 3), (1, 2, 6, 5), # left and right faces
            (0, 1, 5, 4), (2, 3, 7, 6)], **kwargs)# front and back faces
        self.localFace2edge = bm.array([
            (3,  2, 1, 0), (8, 9, 10, 11),
            (4, 11, 7, 3), (1, 6,  9,  5),
            (0,  5, 8, 4), (2, 7, 10,  6)], **kwargs)
        self.localEdge2face = bm.array([
            [4, 0], [3, 0], [5, 0], [0, 2],
            [2, 4], [4, 3], [3, 5], [5, 2],
            [1, 4], [1, 3], [1, 5], [2, 1]], **kwargs)

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
        qf = GaussLegendreQuadrature(q, dtype=self.ftype, device=self.device)
        if etype in {'cell', 3}:
            return TensorProductQuadrature((qf, qf, qf))
        elif etype in {'face', 2}:
            return TensorProductQuadrature((qf, qf))
        elif etype in {'edge', 1}:
            return TensorProductQuadrature((qf,))
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def entity_measure(self, etype=3, index=_S):
        if etype in {'cell', 3}:
            return self.cell_volume(index=index)
        elif etype in {'face', 2}:
            return self.face_area(index=index)
        elif etype in {'edge', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return bm.zeros(1, dtype=self.ftype)
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def cell_volume(self, index=_S):
        """
        @brief 计算单元的体积, 体积的计算公式为
            int_c dx = int_tau |J| d xi
            其中 c 为单元，tau 为参考单元，J 为雅克比矩阵
        """
        qf = self.quadrature_formula(2, etype=3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = self.jacobi_matrix(bcs, index=index)
        detJ = bm.linalg.det(J)
        val = bm.einsum('q, cq -> c', ws, detJ)
        return val

    def face_area(self, index=_S):
        """
        @brief 计算面的面积, 面积的计算公式为
                        int_f ds = int_tau |J| d xi
               其中 f 为面，tau 为参考面，J 为雅克比矩阵
        """
        qf = self.quadrature_formula(2, etype=2)
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = self.jacobi_matrix(bcs, index=index)
        n = bm.cross(J[..., 0], J[..., 1], axis=-1)
        n = bm.sqrt(bm.sum(n**2, axis=-1))
        val = bm.einsum('q, qi->i', ws, n)
        return val

    def jacobi_matrix(self, bc, index=_S):
        """
        @brief 计算参考实体到实际实体间映射的 Jacobi 矩阵
            x(u, v, w) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        assert isinstance(bc, tuple)
        TD = len(bc)
        node = self.entity('node')
        entity = self.entity(TD, index=index)
        gphi = self.grad_shape_function(bc, p=1, variables='u')          # (NQ, ldof, GD)
        if TD == 3:
            node_cell_flip = node[entity[:, [0, 4, 3, 7, 1, 5, 2, 6]]]   # (NC, NCN, GD)
            J = bm.einsum( 'cim, qin -> cqmn', node_cell_flip, gphi)
        elif TD == 2:
            J = bm.einsum( 'cim, qin -> cqmn', node[entity[:, [0, 3, 1, 2]]], gphi)
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

    def interpolation_points(self, p, index=_S):
        """
        @brief Generate interpolation points for the entire mesh
        """
        cell = self.entity('cell')

        c2ip = self.cell_to_ipoint(p)
        gp = self.number_of_global_ipoints(p)
        ipoint = bm.zeros([gp, 3], dtype=self.ftype, device=bm.get_device(cell))

        line = (bm.linspace(0, 1, p+1, endpoint=True,
                        dtype=self.ftype, device=bm.get_device(cell))).reshape(-1, 1)
        line = bm.concatenate([1-line, line], axis=1)
        bcs = (line, line, line)

        cip = self.bc_to_point(bcs)
        ipoint[c2ip] = cip

        return ipoint

    def face_to_ipoint(self, p, index=_S):
        """
        @brief 生成每个面上的插值点全局编号
        """
        return self.quad_to_ipoint(p, index)

    def cell_to_ipoint(self, p, index=_S):
        """!
        @brief Generate global indices for interpolation points in each cell
        """
        cell = self.entity('cell', index=index)
        if p == 1:
            return cell[:, [0, 4, 3, 7, 1, 5, 2, 6]]

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        NC = self.number_of_cells()

        cell2face = self.cell_to_face()
        face2edge = self.face_to_edge()
        cell2edge = self.cell_to_edge()

        face2ipoint = self.face_to_ipoint(p)

        shape = (p+1, p+1, p+1)
        mi = bm.arange(p+1, device=bm.get_device(cell))
        multiIndex0 = bm.broadcast_to(mi[:, None, None], shape).reshape(-1, 1)
        multiIndex1 = bm.broadcast_to(mi[None, :, None], shape).reshape(-1, 1)
        multiIndex2 = bm.broadcast_to(mi[None, None, :], shape).reshape(-1, 1)

        multiIndex = bm.concatenate([multiIndex0, multiIndex1, multiIndex2], axis=-1)

        dofidx = bm.zeros((6, (p+1)**2),
                        dtype=self.itype, device=bm.get_device(cell))
        dofidx[0], = bm.nonzero(multiIndex[:, 2]==0)
        dofidx[1], = bm.nonzero(multiIndex[:, 2]==p)
        dofidx[2], = bm.nonzero(multiIndex[:, 0]==0)
        dofidx[3], = bm.nonzero(multiIndex[:, 0]==p)
        dofidx[4], = bm.nonzero(multiIndex[:, 1]==0)
        dofidx[5], = bm.nonzero(multiIndex[:, 1]==p)

        cell2ipoint = bm.zeros([NC, (p+1)**3],
                            dtype=self.itype, device=bm.get_device(cell))
        lf2e = bm.array([[0, 1, 2, 3], [8, 9, 10, 11],
                         [3, 7, 11, 4], [1, 6, 9, 5],
                         [0, 5, 8, 4], [2, 6, 10, 7]], dtype=self.itype)

        multiIndex2d = multiIndex[:(p+1)**2, 1:]
        multiIndex2d = bm.concatenate([multiIndex2d, p-multiIndex2d], axis=-1)

        lf2e = lf2e[:, [3, 0, 1, 2]]
        face2edge = face2edge[:, [3, 0, 1, 2]]
        for i in range(6):
            gfe = face2edge[cell2face[:, i]]
            lfe = cell2edge[:, lf2e[i]]
            idx0 = bm.argsort(gfe, axis=-1)
            idx1 = bm.argsort(lfe, axis=-1)
            idx1 = bm.argsort(idx1, axis=-1)
            idx0 = idx0[bm.arange(NC)[:, None], idx1] #(NC, 4)
            idx = multiIndex2d[:, idx0].swapaxes(0, 1) #(NC, NQ, 4)

            idx = idx[..., 0]*(p+1)+idx[..., 1]
            cell2ipoint = bm.set_at(cell2ipoint, (slice(None), dofidx[i]),
                                    face2ipoint[cell2face[:, i, None], idx])
            # cell2ipoint[:, dofidx[i]] = face2ipoint[cell2face[:, i, None], idx]

        indof = bm.all(multiIndex>0, axis=-1) & bm.all(multiIndex<p, axis=-1)
        cell2ipoint = bm.set_at(cell2ipoint, (slice(None), indof),
                        bm.arange(NN + NE*(p-1) + NF*(p-1)**2, NN + NE*(p-1) + NF*(p-1)**2 + NC*(p-1)**3,
                        dtype=cell2ipoint.dtype, device=bm.get_device(cell2ipoint)).reshape(NC, -1))
        # cell2ipoint[:, indof] = bm.arange(NN+NE*(p-1)+NF*(p-1)**2,
        #         NN+NE*(p-1)+NF*(p-1)**2+NC*(p-1)**3).reshape(NC, -1)

        return cell2ipoint[index]

    def uniform_refine(self, n=1, surface=None, interface=None, returnim=False):
        """
        @brief Uniform refine the hexahedron mesh n times.

        Parameters:
            n (int): times refine the hexahedron mesh.
            surface (function): the surface function.
            returnim (bool): return the interpolation matrix or not.
        """
        if returnim is True:
            IM = []
        for i in range(n):
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NF = self.number_of_faces()
            NC = self.number_of_cells()
            node = bm.zeros((NN + NE + NF + NC, 3),
                            dtype=self.ftype, device=self.device)
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

            cell = bm.zeros((8*NC, 8),
                            dtype=self.itype, device=self.device)
            c2n = self.entity('cell')
            c2e = self.cell_to_edge() + NN
            c2f = self.cell_to_face() + (NN + NE)
            c2c = bm.arange(NC, device=bm.get_device(cell)) + (NN + NE + NF)
            edge2node = self.edge_to_node()
            face2node = self.face_to_node()
            cell2node = self.cell_to_node()
            if returnim is True:
                new_node_num = NN+NE+NF+NC
                nonzeros = NN+2*NE+4*NF+8*NC
                data = bm.zeros(nonzeros,dtype=bm.float64)
                data[:NN] = 1
                data[NN:NN+2*NE] = 1/2
                data[NN+2*NE:NN+2*NE+4*NE] = 1/4
                data[NN+2*NE+4*NF:] = 1/8

                indices = bm.zeros(nonzeros,dtype=bm.int32)
                indices[:NN] = bm.arange(NN)
                indices[NN:NN+2*NE] = edge2node.flatten()
                indices[NN+2*NE:NN+2*NE+4*NF] = face2node.flatten()
                indices[NN+2*NE+4*NF:] = cell2node.flatten()

                indptr = bm.zeros(new_node_num+1,dtype=bm.int32)
                indptr[:NN+1] = bm.arange(NN+1)
                indptr[NN+1:NN+NE+1]=bm.arange(NN+2,NN+2*NE+1,step=2)
                indptr[NN+NE+1:NN+NE+NF+1] = bm.arange(NN+2*NE+4,NN+2*NE+4*NF+1,step=4)
                indptr[NN+NE+NF+1:] = bm.arange(NN+2*NE+4*NF+8,nonzeros+1,step=8)

                A = csr_matrix((data,indices,indptr),dtype=bm.float64)
                IM.append(A)

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
            self.cell = cell
            self.construct()

        if returnim is True:
            return IM


    @classmethod
    def from_one_hexahedron(cls, twist=False):
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

        if twist:
            upnode = node[4:]
            upnode -= bm.array([[0.5, 0.5, 1]], dtype=bm.float64)
            upnode = bm.cross(bm.array([[0.0, 0.0, 1.0]], dtype=bm.float64), upnode, axis=1)
            node[4:] = upnode + bm.array([[0.5, 0.5, 1]], dtype=bm.float64)

        cell = bm.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=bm.int32)
        return cls(node, cell)

    @classmethod
    def from_box(cls, box=[0, 1, 0, 1, 0, 1], nx=10, ny=10, nz=10,
                threshold=None, *, itype=None, ftype=None, device=None,):
        """
        Generate a hexahedral mesh for a box domain.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param nz Number of divisions along the z-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return HexahedronMesh instance
        """
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64

        shape = (nx+1, ny+1, nz+1)
        X = bm.linspace(box[0], box[1], nx+1, endpoint=True, dtype=ftype, device=device)[:, None, None]
        Y = bm.linspace(box[2], box[3], ny+1, endpoint=True, dtype=ftype, device=device)[None, :, None]
        Z = bm.linspace(box[4], box[5], nz+1, endpoint=True, dtype=ftype, device=device)[None, None, :]
        X = bm.broadcast_to(X, shape).reshape(-1, 1)
        Y = bm.broadcast_to(Y, shape).reshape(-1, 1)
        Z = bm.broadcast_to(Z, shape).reshape(-1, 1)

        node = bm.concatenate([X, Y, Z], axis=-1)

        NN = (nx+1)*(ny+1)*(nz+1)
        idx = bm.arange(0, NN, dtype=itype, device=device).reshape(nx+1, ny+1, nz+1)
        c = idx[:-1, :-1, :-1]

        nyz = (ny + 1)*(nz + 1)
        cell0 = c.reshape(-1, 1)
        cell1 = cell0 + nyz
        cell2 = cell1 + nz + 1
        cell3 = cell0 + nz + 1
        cell4 = cell0 + 1
        cell5 = cell4 + nyz
        cell6 = cell5 + nz + 1
        cell7 = cell4 + nz + 1

        cell = bm.concatenate([cell0, cell1, cell2, cell3, cell4, cell5, cell6, cell7], axis=-1)

        if threshold is not None:
            bc = bm.sum(node[cell, :], axis=1)/cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = bm.zeros(NN, dtype=bm.bool, device=device)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype, device=device)
            idxMap[isValidNode] = bm.arange(isValidNode.sum(), dtype=itype, device=device)
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
        node = bm.array(node_coords, dtype=bm.float64).reshape(-1, 3)

        #节点的编号映射
        nodetags_map = dict({j:i for i,j in enumerate(node_tags)})

        # 获取四面体单元信息
        Hexahedron_type = 5
        Hexahedron_tags, Hexahedron_connectivity = gmsh.model.mesh.getElementsByType(Hexahedron_type)
        evid = bm.array([nodetags_map[j] for j in Hexahedron_connectivity])
        cell = evid.reshape((Hexahedron_tags.shape[-1],-1))

        gmsh.finalize()
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of cells: {cell.shape[0]}")

        return cls(node,cell)

    @classmethod
    def from_crack_box(cls, box=[0, 2, 0, 5, 0, 10], nx=2, ny=5, nz=10,
                       threshold=None, itype=None, ftype=None, device=None):
        """
        Generate a tetrahedral mesh for a box domain.

        @param nx Number of divisions along the x-axis (default: 2)
        @param ny Number of divisions along the y-axis (default: 5)
        @param nz Number of divisions along the z-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TetrahedronMesh instance
        """
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64
        temp_mesh = cls.from_box(box=box, nx=nx, ny=ny, nz=nz, threshold=threshold, itype=itype, ftype=ftype, device=device)
        node = temp_mesh.node
        cell = temp_mesh.cell

        # 切口节点重复
        NN = node.shape[0]
        # 找到切口处 node
        nidx = bm.nonzero((bm.abs(node[:, 2] - 5) < 1e-5) & (node[:, 1] > 3.01))[0]
        # 找到切口处节点所在单元
        nidxmap = bm.arange(NN, dtype=itype, device=device)

        nidxmap = bm.set_at(nidxmap, nidx, NN + bm.arange(len(nidx), dtype=itype, device=device))
        # 计算 z 坐标平均值
        flag = bm.mean(node[:, 2][cell], axis=1) > 5
        cell = bm.set_at(cell, flag, nidxmap[cell[flag]])

        node = bm.concatenate((node, node[nidx]), axis=0)
        mesh = cls(node, cell)
        return mesh

    @classmethod
    def from_seven_hex_cube(cls, itype=None, ftype=None, device=None):
        """
        @brief 构造一个只有七个六面体的网格
        """
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64

        node = bm.array([[0.249, 0.342, 0.192],
                         [0.826, 0.288, 0.288],
                         [0.850, 0.649, 0.263],
                         [0.273, 0.750, 0.230],
                         [0.320, 0.186, 0.643],
                         [0.677, 0.305, 0.683],
                         [0.788, 0.693, 0.644],
                         [0.165, 0.745, 0.702],
                         [0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1]],
                        dtype=ftype, device=device)

        cell = bm.array([[0, 1, 2, 3, 4, 5, 6, 7],
                         [0, 3, 2, 1, 8, 11, 10, 9],
                         [4, 5, 6, 7, 12, 13, 14, 15],
                         [3, 7, 6, 2, 11, 15, 14, 10],
                         [0, 1, 5, 4, 8, 9, 13, 12],
                         [1, 2, 6, 5, 9, 10, 14, 13],
                         [0, 4, 7, 3, 8, 12, 15, 11]],
                        dtype=itype, device=device)
        return cls(node, cell)

    def to_vtk(self, fname=None, etype='cell', index:Index=_S):
        from .vtk_extent import  write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()

        cell = self.entity(etype)[index]
        NC = len(cell)
        NV = cell.shape[-1]

        cell = bm.concatenate((bm.zeros((len(cell), 1), dtype=cell.dtype), cell), axis=1)
        cell[:, 0] = NV

        if etype == 'cell':
            cellType = 12  # 六面体
            celldata = self.celldata
        elif etype == 'face':
            cellType = 5  # 三角形
            celldata = self.facedata
        elif etype == 'edge':
            cellType = 3  # segment
            celldata = self.edgedata

        if fname is None:
            return node, cell.flatten(), cellType, NC
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=celldata)


HexahedronMesh.set_ploter('3d')
