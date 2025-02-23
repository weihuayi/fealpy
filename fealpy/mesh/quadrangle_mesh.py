from typing import Union, Optional, List, Tuple, Any, Callable

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from .. import logger
from .utils import estr2dim

from .mesh_base import TensorMesh
from .plot import Plotable

from fealpy.sparse import coo_matrix,csr_matrix


class QuadrangleMesh(TensorMesh, Plotable):
    def __init__(self, node, cell):
        """
        """
        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)
        kwargs = bm.context(cell)
        self.node = node
        self.cell = cell
        self.localEdge = bm.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], **kwargs)
        self.localFace = bm.tensor([(0, 1), (1, 2), (2, 3), (3, 0)], **kwargs)
        self.ccw = bm.tensor([0, 1, 2, 3], **kwargs)

        self.localCell = None

        self.construct()

        self.meshtype = 'quad'
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.celldata = {}
        self.meshdata = {}

        self.edge_bc_to_point = self.bc_to_point
        self.face_bc_to_point = self.bc_to_point
        self.cell_bc_to_point = self.bc_to_point

        self.cell_grad_shape_function = self.grad_shape_function
        self.cell_shape_function = self.shape_function

        self.face_normal = self.edge_normal
        self.face_unit_normal = self.edge_unit_normal

        self.face_to_ipoint = self.edge_to_ipoint

    def ref_cell_measure(self):
        return 1.0

    def ref_face_measure(self):
        return 1.0

    def cell_area(self, index: Index = _S) -> TensorLike:
        """
        @brief 计算多边形的面积
        """
        GD = self.GD
        if GD == 2:
            node = self.entity('node')
            cell = self.entity('cell')[index]

            v0 = node[cell[:, 1]] - node[cell[:, 0]]
            v1 = node[cell[:, 2]] - node[cell[:, 0]]
            v2 = node[cell[:, 3]] - node[cell[:, 0]]

            # Compute the 2D cross product manually (for z-component)
            cross_01 = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
            cross_12 = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

            s1 = 0.5 * bm.abs(cross_01)
            s2 = 0.5 * bm.abs(cross_12)
            s = s1 + s2

            return s
        elif GD == 3:
            node = self.entity('node')
            cell = self.entity('cell')[index]

            v0 = node[cell[:, 1]] - node[cell[:, 0]]
            v1 = node[cell[:, 2]] - node[cell[:, 0]]
            v2 = node[cell[:, 3]] - node[cell[:, 0]]

            s1 = 0.5 * bm.linalg.norm(bm.cross(v0, v1), axis=-1)
            s2 = 0.5 * bm.linalg.norm(bm.cross(v1, v2), axis=-1)
            s = s1 + s2
            return s

    def entity_measure(self, etype: Union[int, str] = 'cell', index: Index = _S) -> TensorLike:
        node = self.node

        if isinstance(etype, str):
            etype = estr2dim(self, etype)

        if etype == 0:
            return bm.tensor([0, ], dtype=self.ftype)
        elif etype == 1:
            edge = self.entity(1, index)
            return bm.edge_length(edge, node)
        elif etype == 2:
            return self.cell_area(index=index)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

    def quadrature_formula(self, q, etype: Union[int, str] = 'cell', qtype=None):
        from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        qf = GaussLegendreQuadrature(q, dtype=self.ftype, device=self.device)
        if etype == 2:
            return TensorProductQuadrature((qf, qf))
        elif etype == 1:
            return TensorProductQuadrature((qf,))
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def jacobi_matrix(self, bc, index: Index = _S) -> TensorLike:
        """
        @brief 计算参考单元 (xi, eta) 到实际 Lagrange 四边形(x) 之间映射的 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        gphi = self.grad_shape_function(bc, p=1, variables='u', index=index) # (NQ, ldof, GD)
        node_cell_flip = node[cell[:, [0, 3, 1, 2]]]
        J = bm.einsum('cim, qin -> cqmn', node_cell_flip, gphi) # (NC, NQ, GD, GD)
        
        return J

    def first_fundamental_form(self, J: TensorLike) -> TensorLike:
        """
        @brief 由 Jacobi 矩阵计算第一基本形式。
        """
        TD = J.shape[-1]
        shape = J.shape[0:-2] + (TD, TD)
        data = [[0 for i in range(TD)] for j in range(TD)]
        for i in range(TD):
            data[i][i] = bm.einsum('...d, ...d->...', J[..., i], J[..., i])
            for j in range(i + 1, TD):
                data[i][j] = bm.einsum('...d, ...d->...', J[..., i], J[..., j])
                data[j][i] = data[i][j]
        data = [val.reshape(val.shape + (1,)) for data_ in data for val in data_]
        G = bm.concatenate(data, axis=-1).reshape(shape)
        return G

    def edge_unit_tangent(self, index: Index = _S) -> TensorLike:
        return self.edge_tangent(index=index, unit=True)

    def edge_unit_normal(self, index: Index = _S) -> TensorLike:
        """
        @brief 计算二维网格中每条边上单位法线
        """
        return self.edge_normal(index=index, unit=True)

    def edge_frame(self, index: Index = _S):
        """
        @brief 计算二维网格中每条边上的局部标架
        """
        assert self.GD == 2
        t = self.edge_unit_tangent(index=index)
        w = bm.tensor([(0, -1), (1, 0)], dtype=t.dtype)
        n = t @ w
        return n, t

    def interpolation_points(self, p: int, index: Index = _S):
        """
        @brief Get all p-th order interpolation points on the quadrilateral mesh
        """
        cell = self.entity('cell')
        node = self.entity('node')
        if p == 1:
            return node[index]

        GD = self.geo_dimension()

        edge = self.entity('edge')

        multiIndex = self.multi_index_matrix(p, 1, dtype=self.ftype, device=bm.get_device(cell))
        w = multiIndex[1:-1, :] / p
        ipoints0 = bm.einsum('ij, ...jm->...im', w, node[edge, :]).reshape(-1, GD)

        w = bm.einsum('im, jn->ijmn', w, w).reshape(-1, 4)
        ipoints1 = bm.einsum('ij, kj...->ki...', w, node[cell[:, [0, 3, 1, 2]]]).reshape(-1, GD)

        ipoints = bm.concatenate((node, ipoints0, ipoints1), axis=0)

        return ipoints[index]

    def number_of_corner_nodes(self):
        return self.number_of_nodes()

    def cell_to_ipoint(self, p: int, index: Index = _S):
        """
        @brief 获取单元上的双 p 次插值点
        """

        cell = self.entity('cell')

        if p == 0:
            return bm.arange(len(cell)).reshape((-1, 1))[index]

        if p == 1:
            return cell[index, [0, 3, 1, 2]]  # 先排 y 方向，再排 x 方向

        edge2cell = self.edge2cell
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        cell2ipoint = bm.zeros((NC, (p + 1) * (p + 1)), dtype=self.itype, device=bm.get_device(cell))
        c2p = cell2ipoint.reshape((NC, p + 1, p + 1))
        e2p = self.edge_to_ipoint(p)

        flag = edge2cell[:, 2] == 0
        c2p = bm.set_at(c2p, (edge2cell[flag, 0], slice(None), 0), e2p[flag])

        flag = edge2cell[:, 2] == 1
        c2p = bm.set_at(c2p, (edge2cell[flag, 0], -1, slice(None)), e2p[flag])

        flag = edge2cell[:, 2] == 2
        c2p = bm.set_at(c2p, (edge2cell[flag, 0], slice(None), -1), bm.flip(e2p[flag], axis=-1))

        flag = edge2cell[:, 2] == 3
        c2p = bm.set_at(c2p, (edge2cell[flag, 0], 0, slice(None)), bm.flip(e2p[flag], axis=-1))

        iflag = edge2cell[:, 0] != edge2cell[:, 1]
        flag = iflag & (edge2cell[:, 3] == 0)
        c2p = bm.set_at(c2p, (edge2cell[flag, 1], slice(None), 0), bm.flip(e2p[flag], axis=-1))

        flag = iflag & (edge2cell[:, 3] == 1)
        c2p = bm.set_at(c2p, (edge2cell[flag, 1], -1, slice(None)), bm.flip(e2p[flag], axis=-1))

        flag = iflag & (edge2cell[:, 3] == 2)
        c2p = bm.set_at(c2p, (edge2cell[flag, 1], slice(None), -1), e2p[flag])

        flag = iflag & (edge2cell[:, 3] == 3)
        c2p = bm.set_at(c2p, (edge2cell[flag, 1], 0, slice(None)), e2p[flag])

        c2p = bm.set_at(c2p, (slice(None), slice(1, -1), slice(1, -1)), NN + NE * (p - 1) +
                        bm.arange(NC * (p - 1) * (p - 1)).reshape(NC, p - 1, p - 1))

        # if bm.backend_name in ["numpy", "pytorch"]:
        #     flag = edge2cell[:, 2] == 0
        #     c2p[edge2cell[flag, 0], :, 0] = e2p[flag]
        #     flag = edge2cell[:, 2] == 1
        #     c2p[edge2cell[flag, 0], -1, :] = e2p[flag]
        #     flag = edge2cell[:, 2] == 2
        #     c2p[edge2cell[flag, 0], :, -1] = bm.flip(e2p[flag], axis=-1)
        #     flag = edge2cell[:, 2] == 3
        #     c2p[edge2cell[flag, 0], 0, :] = bm.flip(e2p[flag], axis=-1)

        #     iflag = edge2cell[:, 0] != edge2cell[:, 1]
        #     flag = iflag & (edge2cell[:, 3] == 0)
        #     c2p[edge2cell[flag, 1], :, 0] = bm.flip(e2p[flag], axis=-1)
        #     flag = iflag & (edge2cell[:, 3] == 1)
        #     c2p[edge2cell[flag, 1], -1, :] = bm.flip(e2p[flag], axis=-1)
        #     flag = iflag & (edge2cell[:, 3] == 2)
        #     c2p[edge2cell[flag, 1], :, -1] = e2p[flag]
        #     flag = iflag & (edge2cell[:, 3] == 3)
        #     c2p[edge2cell[flag, 1], 0, :] = e2p[flag]

        #     c2p[:, 1:-1, 1:-1] = NN + NE * (p - 1) + bm.arange(NC * (p - 1) * (p - 1)).reshape(NC, p - 1, p - 1)
        # elif bm.backend_name == "jax":
        #     flag = edge2cell[:, 2] == 0
        #     c2p = c2p.at[edge2cell[flag, 0], :, 0].set(e2p[flag])
        #     flag = edge2cell[:, 2] == 1
        #     c2p = c2p.at[edge2cell[flag, 0], -1, :].set(e2p[flag])
        #     flag = edge2cell[:, 2] == 2
        #     c2p = c2p.at[edge2cell[flag, 0], :, -1].set(e2p[flag, -1::-1])
        #     flag = edge2cell[:, 2] == 3
        #     c2p = c2p.at[edge2cell[flag, 0], 0, :].set(e2p[flag, -1::-1])

        #     iflag = edge2cell[:, 0] != edge2cell[:, 1]
        #     flag = iflag & (edge2cell[:, 3] == 0)
        #     c2p = c2p.at[edge2cell[flag, 1], :, 0].set(e2p[flag, -1::-1])
        #     flag = iflag & (edge2cell[:, 3] == 1)
        #     c2p = c2p.at[edge2cell[flag, 1], -1, :].set(e2p[flag, -1::-1])
        #     flag = iflag & (edge2cell[:, 3] == 2)
        #     c2p = c2p.at[edge2cell[flag, 1], :, -1].set(e2p[flag])
        #     flag = iflag & (edge2cell[:, 3] == 3)
        #     c2p = c2p.at[edge2cell[flag, 1], 0, :].set(e2p[flag])

        #     c2p = c2p.at[:, 1:-1, 1:-1].set(NN + NE * (p - 1) + bm.arange(NC * (p - 1) * (p - 1)).reshape(NC, p - 1, p - 1))

        #     cell2ipoint = c2p.reshape((NC, (p + 1) * (p + 1)))
        # else:
        #     raise ValueError("Unsupported backend")
        return cell2ipoint[index]
    
    def prolongation_matrix(self, p0: int, p1: int):
        """
        Return the prolongation_matrix from p0 to p1: 0 < p0 < p1

        Parameters:
            p0(int): The degree of the lowest-order space.
            p1(int): The degree of the highest-order space.

        Returns:
            CSRTensor: the prolongation_matrix from p0 to p1
        """
        assert 0 < p0 < p1

        TD = self.top_dimension()#Geometric Dimension
        gdof0 = self.number_of_global_ipoints(p0)
        gdof1 = self.number_of_global_ipoints(p1)
        matrix_shape = (gdof1,gdof0)

        # 1. Interpolation points on the mesh nodes: Inherit the original interpolation points
        NN = self.number_of_nodes()
        V_1 = bm.ones(NN)
        I_1 = bm.arange(NN)
        J_1 = bm.arange(NN)

        # 2. Interpolation points within the mesh edges
        NE = self.number_of_edges()
        bcs = self.multi_index_matrix(p1, TD - 1) / p1

        phi = self.edge_shape_function(bcs=(bcs[1:-1],), p=p0)  # (ldof1 - 2, ldof0)

        e2p1 = self.edge_to_ipoint(p1)[:, 1:-1]
        e2p0 = self.edge_to_ipoint(p0)
        shape = (NE,) + phi.shape

        I_2 = bm.broadcast_to(e2p1[:, :, None], shape=shape).flat
        J_2 = bm.broadcast_to(e2p0[:, None, :], shape=shape).flat
        V_2 = bm.broadcast_to(phi[None, :, :], shape=shape).flat

        # 3. Interpolation points within the mesh cells
        NC = self.number_of_cells()
        bcs = self.multi_index_matrix(p1, TD -1) / p1
        phi = self.cell_shape_function((bcs[1:-1],bcs[1:-1]), p=p0)
        c2p1 = self.cell_to_ipoint(p1).reshape(NC,p1+1,p1+1)[:,1:-1,1:-1]
        c2p1 = c2p1.reshape(NC,-1)
        c2p0 = self.cell_to_ipoint(p0)
        shape = (NC,) + phi.shape

        I_3 = bm.broadcast_to(c2p1[:, :, None], shape=shape).flat
        J_3 = bm.broadcast_to(c2p0[:, None, :], shape=shape).flat
        V_3 = bm.broadcast_to(phi[None, :, :], shape=shape).flat
       
        # 4.concatenate
        V = bm.concatenate((V_1, V_2, V_3), axis=0) 
        I = bm.concatenate((I_1, I_2, I_3), axis=0) 
        J = bm.concatenate((J_1, J_2, J_3), axis=0) 
        P = csr_matrix((V, (I, J)), matrix_shape)

        return P

    def jacobi_at_corner(self) -> TensorLike:
        NC = self.number_of_cells()
        node = self.entity('node')
        cell = self.entity('cell')
        localEdge = self.localEdge
        iprev = [3, 0, 1, 2]
        jacobis = []
        for i, j in localEdge:
            k = iprev[i]
            v0 = node[cell[:, j], :] - node[cell[:, i], :]
            v1 = node[cell[:, k], :] - node[cell[:, i], :]
            jacobis.append((v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]).reshape(-1, 1))
        jacobi = bm.concatenate(jacobis, axis=-1)
        return jacobi

    def angle(self) -> TensorLike:
        NC = self.number_of_cells()
        node = self.node
        cell = self.cell
        localEdge = self.localEdge
        iprev = [3, 0, 1, 2]
        angles = []
        for i, j in localEdge:
            k = iprev[i]
            v0 = node[cell[:, j], :] - node[cell[:, i], :]
            v1 = node[cell[:, k], :] - node[cell[:, i], :]
            angles.append(bm.arccos(
                bm.sum(v0 * v1, axis=1)
                / bm.sqrt(bm.sum(v0 ** 2, axis=1)
                          * bm.sum(v1 ** 2, axis=1))).reshape(-1, 1))
        angle = bm.concatenate(angles, axis=-1)
        return angle

    def cell_quality(self) -> TensorLike:
        jacobi = self.jacobi_at_corner()
        return jacobi.sum(axis=1) / 4

    def reorder_cell(self, idx):
        raise NotImplementedError
        # NC = self.number_of_cells()
        # NN = self.number_of_nodes()
        # cell = self.cell
        # # localCell 似乎之前未初始化
        # cell = cell[bm.arange(NC).reshape(-1, 1), self.localCell[idx]]
        # self.ds.reinit(NN, cell)

    def uniform_refine(self, n=1, surface=None, interface=None, returnim=False) -> 'QuadrangleMesh':
        """
        Uniform refine the quadrangle mesh n times.

        Parameters:
            n (int): Times refine the triangle mesh.
            surface (function): The surface function.
            returnirm (bool): Return the prolongation matrix list or not,from the finest to the the coarsest
        
        Returns:
            mesh: The mesh obtained after uniformly refining n times.
            List(CSRTensor): The prolongation matrix from the finest to the the coarsest
        """
        if returnim is True:
            IM = []
        for i in range(n):
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()
            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')            
            edge2node = self.edge_to_node()
            cell2node = self.cell_to_node()

            if returnim is True:
                shape = (NN+NE+NC,NN)
                kargs = bm.context(node)
                values = bm.ones(NN+2*NE+4*NC,**kargs)
                values = bm.set_at(values,bm.arange(NN, NN+2*NE), 0.5)
                values = bm.set_at(values,bm.arange(NN+2*NE,NN+2*NE+4*NC), 0.25)
                
                kargs = bm.context(cell)
                i0 = bm.arange(NN,**kargs)
                i1 = bm.arange(NN, NN + NE, **kargs)
                i2 = bm.arange(NN+NE, NN + NE + NC, **kargs)
                I = bm.concatenate((i0,i1,i1,i2,i2,i2,i2))
                J = bm.concatenate((i0,edge[:,0],edge[:,1],cell[:,0],cell[:,1],cell[:,2],cell[:,3]))

                P = csr_matrix((values,(I,J)),shape)

                IM.append(P)

            # Find the cutted edge
            cell2edge = self.cell2edge
            edgeCenter = self.entity_barycenter('edge')
            cellCenter = self.entity_barycenter('cell')

            edge2center = bm.arange(NN, NN + NE, device=bm.get_device(cell2edge))

            cell = self.cell
            cp = [cell[:, i].reshape(-1, 1) for i in range(4)]
            ep = [edge2center[cell2edge[:, i]].reshape(-1, 1) for i in range(4)]
            cc = bm.arange(NN + NE, NN + NE + NC, device=bm.get_device(cell2edge)).reshape(-1, 1)

            cell = bm.zeros((4 * NC, 4), dtype=self.itype, device=bm.get_device(cell2edge))

            cell = bm.set_at(cell, (slice(0, None, 4), slice(None)),
                             bm.concatenate([cp[0], ep[0], cc, ep[3]], axis=1))
            cell = bm.set_at(cell, (slice(1, None, 4), slice(None)),
                             bm.concatenate([ep[0], cp[1], ep[1], cc], axis=1))
            cell = bm.set_at(cell, (slice(2, None, 4), slice(None)),
                             bm.concatenate([cc, ep[1], cp[2], ep[2]], axis=1))
            cell = bm.set_at(cell, (slice(3, None, 4), slice(None)),
                             bm.concatenate([ep[3], cc, ep[2], cp[3]], axis=1))
            # cell[0::4, :] = bm.concatenate([cp[0], ep[0], cc, ep[3]], axis=1)
            # cell[1::4, :] = bm.concatenate([ep[0], cp[1], ep[1], cc], axis=1)
            # cell[2::4, :] = bm.concatenate([cc, ep[1], cp[2], ep[2]], axis=1)
            # cell[3::4, :] = bm.concatenate([ep[3], cc, ep[2], cp[3]], axis=1)

            # if bm.backend_name in ["numpy", "pytorch"]:
            #     cell[0::4, :] = bm.concatenate([cp[0], ep[0], cc, ep[3]], axis=1)
            #     cell[1::4, :] = bm.concatenate([ep[0], cp[1], ep[1], cc], axis=1)
            #     cell[2::4, :] = bm.concatenate([cc, ep[1], cp[2], ep[2]], axis=1)
            #     cell[3::4, :] = bm.concatenate([ep[3], cc, ep[2], cp[3]], axis=1)
            # elif bm.backend_name == "jax":
            #     row_indices = bm.arange(4 * NC, device=bm.get_device(cell2edge)).reshape(NC, 4)
            #     # 将单元块的行索引和列索引拼接在一起
            #     cell = cell.at[row_indices[:, 0]].set(bm.concatenate([cp[0], ep[0], cc, ep[3]], axis=-1))
            #     cell = cell.at[row_indices[:, 1]].set(bm.concatenate([ep[0], cp[1], ep[1], cc], axis=-1))
            #     cell = cell.at[row_indices[:, 2]].set(bm.concatenate([cc, ep[1], cp[2], ep[2]], axis=-1))
            #     cell = cell.at[row_indices[:, 3]].set(bm.concatenate([ep[3], cc, ep[2], cp[3]], axis=-1))

            # else:
            #     raise ValueError("Unsupported backend")

            self.node = bm.concatenate([self.node, edgeCenter, cellCenter], axis=0)
            self.cell = cell

            self.construct()
        if returnim is True:
            IM.reverse()
            return IM
        
    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 2}:
            VTK_Quad = 9
            return VTK_Quad
        elif etype in {'face', 'edge', 1}:
            VTK_LINE = 3
            return VTK_LINE

    def to_vtk(self, fname=None, etype='cell', index: Index = _S):

        from fealpy.mesh.vtk_extent import write_to_vtu

        node = self.entity('node')
        GD = self.GD
        if GD == 2:
            node = bm.concatenate((node, bm.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        NV = cell.shape[-1]

        cell = bm.concatenate((bm.zeros((len(cell), 1), dtype=cell.dtype), cell), axis=1)
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                         nodedata=self.nodedata,
                         celldata=self.celldata)

    def show_function(self, plot, uh, cmap=None):
        """
        TODO: no test
        """
        from types import ModuleType
        import matplotlib.colors as colors
        import matplotlib.cm as cm
        from mpl_toolkits.mplot3d import Axes3D
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = plot.axes(projection='3d')
        else:
            axes = plot

        node = self.node
        cax = axes.plot_trisurf(
            node[:, 0], node[:, 1],
            uh, cmap=cmap, lw=0.0)
        axes.figure.colorbar(cax, ax=axes)
        return axes

    @classmethod
    def from_box(cls, box=[0, 1, 0, 1], nx=10, ny=10,
                 threshold: Optional[Callable] = None, device: str = None) -> 'QuadrangleMesh':
        """
        Generate a quadrilateral mesh for a rectangular domain.
        """
        NN = (nx + 1) * (ny + 1)
        node = bm.zeros((NN, 2), device=device)
        x = bm.linspace(box[0], box[1], nx + 1, dtype=bm.float64, device=device)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=bm.float64, device=device)
        X, Y = bm.meshgrid(x, y, indexing='ij')
        node = bm.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)

        idx = bm.arange(NN, device=device).reshape(nx + 1, ny + 1)
        cell = bm.concatenate((idx[0:-1, 0:-1].reshape(-1, 1),
                               idx[1:, 0:-1].reshape(-1, 1),
                               idx[1:, 1:].reshape(-1, 1),
                               idx[0:-1, 1:].reshape(-1, 1),), axis=1)

        if threshold is not None:
            bc = bm.sum(node[cell, :], axis=1) / cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = bm.zeros(NN, dtype=bm.bool)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype, device=device)
            idxMap = bm.set_at(idxMap, isValidNode, bm.arange(isValidNode.sum()))
            idxMap[isValidNode] = bm.arange(isValidNode.sum(), device=device)
            cell = idxMap[cell]

        return cls(node, cell)

    @classmethod
    def from_unit_square(cls, nx=10, ny=10, threshold: Optional[Callable] = None) -> 'QuadrangleMesh':
        """
        Generate a quadrilateral mesh for a unit square.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return QuadrangleMesh instance
        """
        return cls.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny, threshold=threshold)

    @classmethod
    def from_polygon_gmsh(cls, vertices: List[tuple], h: float) -> 'QuadrangleMesh':
        """
        Generate a quadrilateral mesh for a polygonal region by gmsh.

        @param vertices List of tuples representing vertices of the polygon
        @param h Parameter controlling mesh density
        @return QuadrilateralMesh instance
        """
        import gmsh
        gmsh.initialize()
        gmsh.model.add("Polygon")

        # 创建多边形
        lc = h  # 设置网格大小
        polygon_points = []
        for i, vertex in enumerate(vertices):
            point = gmsh.model.geo.addPoint(vertex[0], vertex[1], 0, lc)
            polygon_points.append(point)

        # 添加线段和循环
        lines = []
        for i in range(len(polygon_points)):
            line = gmsh.model.geo.addLine(polygon_points[i], polygon_points[(i + 1) % len(polygon_points)])
            lines.append(line)
        curve_loop = gmsh.model.geo.addCurveLoop(lines)

        # 创建平面表面
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        # 同步几何模型
        gmsh.model.geo.synchronize()

        # 添加物理组
        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        gmsh.model.setPhysicalName(2, 1, "Polygon")

        # 设置网格算法选项，使用 Quadrangle 2D 算法
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        # 生成网格
        gmsh.model.mesh.generate(2)

        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = bm.tensor(node_coords, dtype=bm.float64).reshape(-1, 3)[:, 0:2]

        # 获取四边形单元信息
        quadrilateral_type = 3  # 四边形单元的类型编号为 3
        quad_tags, quad_connectivity = gmsh.model.mesh.getElementsByType(quadrilateral_type)
        cell = bm.tensor(quad_connectivity, dtype=bm.int64).reshape(-1, 4) - 1

        # 输出节点和单元数量
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of quadrilaterals: {cell.shape[0]}")

        gmsh.finalize()

        NN = len(node)
        if bm.backend_name in ["numpy", "pytorch"]:
            isValidNode = bm.zeros(NN, dtype=bm.bool)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = bm.arange(isValidNode.sum())
        elif bm.backend_name == "jax":
            isValidNode = bm.zeros(NN, dtype=bm.bool)
            isValidNode = isValidNode.at[cell].set(True)
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype)
            idxMap = idxMap.at[isValidNode].set(bm.arange(isValidNode.sum()))
        else:
            raise ValueError("Unsupported backend")
        cell = idxMap[cell]

        return cls(node, cell)

    @classmethod
    def from_fuel_rod_gmsh(cls, R1, R2, L, w, h, meshtype='normal'):
        raise NotImplementedError

    @classmethod
    def from_one_quadrangle(cls, meshtype='square') -> 'QuadrangleMesh':
        """
        Generate a quadrilateral mesh for a single quadrangle.

        @param meshtype Type of quadrangle mesh, options are 'square', 'zhengfangxing', 'rectangle', 'rec', 'juxing', 'rhombus', 'lingxing' (default: 'square')
        @return QuadrangleMesh instance
        """
        if meshtype in {'square'}:
            node = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0]], dtype=bm.float64)
        elif meshtype in {'rectangle'}:
            node = bm.tensor([
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0]], dtype=bm.float64)
        elif meshtype in {'rhombus'}:
            import math
            node = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.5, math.sqrt(3) / 2],
                [0.5, math.sqrt(3) / 2]], dtype=bm.float64)
        cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int64)
        return cls(node, cell)

    @classmethod
    def from_square_domain_with_fracture(cls):
        node = bm.tensor([
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0]], dtype=bm.float64)

        cell = bm.tensor([
            [0, 4, 5, 1],
            [4, 7, 8, 5],
            [5, 8, 9, 6],
            [2, 5, 6, 3]], dtype=bm.int32)

        return cls(node, cell)

    @classmethod
    def from_triangle_mesh(cls, mesh) -> 'QuadrangleMesh':
        """
        把每个三角形分成三个四边形
        @param mesh: 三角形网格
        @return:
        """
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        node0 = mesh.node
        cell0 = mesh.cell
        ec = mesh.entity_barycenter('edge')
        cc = mesh.entity_barycenter('cell')
        cell2edge = mesh.cell2edge

        node = bm.concatenate([node0, ec, cc], axis=0)
        idx = bm.arange(NC)

        cell1 = bm.concatenate([(NN + NE + idx).reshape(-1, 1),
                                (cell2edge[:, 0] + NN).reshape(-1, 1),
                                (cell0[:, 2]).reshape(-1, 1),
                                (cell2edge[:, 1] + NN).reshape(-1, 1)], axis=1)
        cell2 = bm.concatenate([(cell1[:, 0]).reshape(-1, 1),
                                (cell2edge[:, 1] + NN).reshape(-1, 1),
                                (cell0[:, 0]).reshape(-1, 1),
                                (cell2edge[:, 2] + NN).reshape(-1, 1)], axis=1)
        cell3 = bm.concatenate([(cell1[:, 0]).reshape(-1, 1),
                                (cell2edge[:, 2] + NN).reshape(-1, 1),
                                (cell0[:, 1]).reshape(-1, 1),
                                (cell2edge[:, 0] + NN).reshape(-1, 1)], axis=1)
        cell = bm.concatenate([cell1, cell2, cell3], axis=0)

        return cls(node, cell)

    @classmethod
    def polygon_domain_generator(cls, num_vertices=20, radius=1.0, center=[0.0, 0.0]):
        raise NotImplementedError

    @classmethod
    def rand_quad_mesh_generator(cls, num, filename=None, h=0.382, radius=0.5, center=[0.5, 0.5]):
        """
        @brief 随机生成指定区域的，指定数量的，随机四边形网格

        @param num: 需要生成的网格数量
        @param filename: 输出文件名，如果非 None，输出文件，如果为 None，返回网格列表
        @param h: 网格密度
        @param radius: 区域半径
        @param center: 区域中点坐标

        @return: None 或网格列表
        """
        raise NotImplementedError

    @classmethod
    def sub_domain_mesh_generator(cls, half_edge, origin_node, separator_streamlines):
        """
        分块四边形网格生成器
        TODO: 考虑是否需要兼容非 numpy 后端
        @param half_edge: 半边数据结构，用于描述分区
        @param origin_node: 各分区节点
        @param separator_streamlines: 各分区边（由边上的点列组成），相对边点数量需一致
        @return: 分块四边形网格
        """
        import numpy as np

        from fealpy.geometry.coons_patch import coons_patch

        # 计算各条边上的点数量，包括两端点
        edge_segments_num = np.array([len(l) for l in separator_streamlines])

        # 构造 cell_to_edge 映射
        cell_num = np.max(half_edge[:, 1]) + 1
        cell_start_edge = np.flip(np.min(np.flip(np.argsort(half_edge[:, 1]))[0:4 * cell_num].reshape((-1, 4)), axis=1))
        cell_to_edge = np.zeros((cell_num, 4), dtype=np.int_)
        for i in range(cell_num):
            next_edge = cell_start_edge[i]
            for j in range(4):
                cell_to_edge[i, j] = next_edge
                next_edge = half_edge[next_edge, 2]

        origin_node_num = len(origin_node)
        node_num = origin_node_num
        # 遍历边界，计算边界内部节点数量，不包括两端点
        edge_node_num = edge_segments_num - 2
        node_num += sum(edge_node_num)
        # 遍历单元，计算单元内部节点数量
        cell_node_num = []
        cell_cell_num = []
        for i in range(cell_num):
            cell_node_num.append(
                (edge_segments_num[cell_to_edge[i, 0] // 2] - 2) * (edge_segments_num[cell_to_edge[i, 1] // 2] - 2))
            cell_cell_num.append(
                (edge_segments_num[cell_to_edge[i, 0] // 2] - 1) * (edge_segments_num[cell_to_edge[i, 1] // 2] - 1))
        node_num += sum(cell_node_num)
        total_cell_num = sum(cell_cell_num)
        # 初始化单元与节点
        node = np.zeros((node_num, 2))
        cell = np.zeros((total_cell_num, 4), dtype=np.int_)
        cell_domain_tag = np.zeros(total_cell_num, dtype=np.int_)
        node[0:origin_node_num] = origin_node
        # 离散边界节点
        edge_node_list = separator_streamlines
        for idx, b_line in enumerate(separator_streamlines):
            node[origin_node_num + sum(edge_node_num[:idx]):origin_node_num + sum(edge_node_num[:idx + 1])] = b_line[
                                                                                                              1:-1]

        for i in range(len(cell_to_edge)):
            edge0 = edge_node_list[cell_to_edge[i, 0] // 2][::((-1) ** (cell_to_edge[i, 0] % 2))]
            edge1 = edge_node_list[cell_to_edge[i, 1] // 2][::((-1) ** (cell_to_edge[i, 1] % 2))]
            edge2 = edge_node_list[cell_to_edge[i, 2] // 2][::((-1) ** (cell_to_edge[i, 2] % 2))]
            edge3 = edge_node_list[cell_to_edge[i, 3] // 2][::((-1) ** (cell_to_edge[i, 3] % 2))]
            n1 = edge_segments_num[cell_to_edge[i, 0] // 2]
            n2 = edge_segments_num[cell_to_edge[i, 1] // 2]
            grid = coons_patch(edge0, edge1, edge2, edge3, n1, n2)
            temp_node = grid[1:-1, 1:-1, :].reshape((-1, 2))
            node[
            origin_node_num + sum(edge_node_num) + sum(cell_node_num[:i]):origin_node_num + sum(edge_node_num) + sum(
                cell_node_num[:i + 1])] = temp_node
            # 子区域单元节点索引构造
            cell_node_idx = np.zeros((n2, n1), dtype=np.int_)
            # 角点
            cell_node_idx[0, 0] = half_edge[cell_to_edge[i, 3], 0]
            cell_node_idx[0, n1 - 1] = half_edge[cell_to_edge[i, 0], 0]
            cell_node_idx[n2 - 1, n1 - 1] = half_edge[cell_to_edge[i, 1], 0]
            cell_node_idx[n2 - 1, 0] = half_edge[cell_to_edge[i, 2], 0]
            # 边
            edge0_idx = cell_to_edge[i, 0] // 2
            edge0_node_idx = np.arange(origin_node_num + sum(edge_node_num[:edge0_idx]),
                                       origin_node_num + sum(edge_node_num[:edge0_idx + 1]))
            cell_node_idx[0, 1:n1 - 1] = edge0_node_idx[::((-1) ** (cell_to_edge[i, 0] % 2))]

            edge1_idx = cell_to_edge[i, 1] // 2
            edge1_node_idx = np.arange(origin_node_num + sum(edge_node_num[:edge1_idx]),
                                       origin_node_num + sum(edge_node_num[:edge1_idx + 1]))
            cell_node_idx[1:n2 - 1, n1 - 1] = edge1_node_idx[::((-1) ** (cell_to_edge[i, 1] % 2))]

            edge2_idx = cell_to_edge[i, 2] // 2
            edge2_node_idx = np.arange(origin_node_num + sum(edge_node_num[:edge2_idx]),
                                       origin_node_num + sum(edge_node_num[:edge2_idx + 1]))
            cell_node_idx[n2 - 1, 1:n1 - 1] = edge2_node_idx[::((-1) ** (cell_to_edge[i, 2] % 2))][::-1]

            edge3_idx = cell_to_edge[i, 3] // 2
            edge3_node_idx = np.arange(origin_node_num + sum(edge_node_num[:edge3_idx]),
                                       origin_node_num + sum(edge_node_num[:edge3_idx + 1]))
            cell_node_idx[1:n2 - 1, 0] = edge3_node_idx[::((-1) ** (cell_to_edge[i, 3] % 2))][::-1]
            # 内部
            cell_node_idx[1:n2 - 1, 1:n1 - 1] = np.arange(origin_node_num + sum(edge_node_num) + sum(cell_node_num[:i]),
                                                          origin_node_num + sum(edge_node_num) + sum(
                                                              cell_node_num[:i + 1])).reshape((n2 - 2, n1 - 2))
            cell[sum(cell_cell_num[:i]):sum(cell_cell_num[:i + 1]), 0] = cell_node_idx[0:-1, 0:-1].flatten()
            cell[sum(cell_cell_num[:i]):sum(cell_cell_num[:i + 1]), 1] = cell_node_idx[0:-1, 1:].flatten()
            cell[sum(cell_cell_num[:i]):sum(cell_cell_num[:i + 1]), 2] = cell_node_idx[1:, 1:].flatten()
            cell[sum(cell_cell_num[:i]):sum(cell_cell_num[:i + 1]), 3] = cell_node_idx[1:, 0:-1].flatten()
            cell_domain_tag[sum(cell_cell_num[:i]):sum(cell_cell_num[:i + 1])] = i

        quad_mesh = cls(node, cell)
        quad_mesh.celldata['cell_domain_tag'] = cell_domain_tag

        return quad_mesh

    @classmethod
    def from_unit_sphere_surface(cls, refine=1, itype=None, ftype=None, device=None):
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64

        def project_to_sphere(nodes):
            norms = bm.sqrt(bm.sum(nodes**2, axis=1, keepdims=True))
            return nodes / norms

        origin_node = bm.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ], dtype=ftype, device=device)
        origin_cell = bm.array(
            [
                [0, 3, 2, 1],
                [4, 5, 6, 7],
                [3, 7, 6, 2],
                [0, 1, 5, 4],
                [0, 4, 7, 3],
                [1, 2, 6, 5]
            ], dtype=itype, device=device)
        origin_node = project_to_sphere(origin_node)
        quad_mesh = cls(origin_node, origin_cell)

        for i in range(refine):
            quad_mesh.uniform_refine()
            refine_node = quad_mesh.node
            refine_node = project_to_sphere(refine_node)
            refine_cell = quad_mesh.cell
            quad_mesh = cls(refine_node, refine_cell)

        return quad_mesh


QuadrangleMesh.set_ploter('2d')
