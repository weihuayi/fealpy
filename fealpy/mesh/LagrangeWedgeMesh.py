import numpy as np

from .Mesh3d import Mesh3d, Mesh3dDataStructure

from ..quadrature import GaussLegendreQuadrature, TriangleQuadrature
from ..quadrature import TensorProductQuadrature

# 单纯形网格的多重指标矩阵
from .core import multi_index_matrix
# 单纯形网格的拉格朗日形函数
from .core import lagrange_shape_function
# 单纯形网格拉格朗日形函数关于重心坐标函数的导数
from .core import lagrange_grad_shape_function

from .core import LinearMeshDataStructure

class LinearWedgeMeshDataStructure(LinearMeshDataStructure):

    localTFace = np.array([(0, 4, 2), (1, 3, 5)])
    localQFace = np.array([(2, 3, 4, 5), (4, 5, 0, 1), (0, 1, 2, 3)])
    localEdge = np.array([
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5), (2, 3),
        (2, 4), (3, 5), (4, 5)])
    localTFace2edge = np.array([(6, 1, 2), (7, 4, 3)])
    localQFace2edge = np.array([(6, 8, 7, 5), (2, 0, 4, 8), (1, 5, 3, 0)])

    V = 6 # 每个单元 6 个顶点 
    E = 9 # 每个单元 9 条边
    F = 5 # 每个单元 5 个面
    EV = 2 # 每个边有 2 个顶点

    def __init__(self, NN, cell):
        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.itype = cell.dtype
        self.construct_edge()
        self.tface, self.tface2cell = self.construct_face(self.localTFace)
        self.qface, self.qface2cell = self.construct_face(self.localQFace)
        self.NTF = len(self.tface)
        self.NQF = len(self.qface)
        self.NF = self.NTF + self.NQF

    def total_face(self, localFace):
        NC = self.NC
        cell = self.cell
        FV = localFace.shape[1] 
        totalFace = cell[:, localFace].reshape(-1, FV)
        return totalFace

    def construct_face(self, localFace):
        """ 

        Notes
        -----
            构造面
        """
        NC = self.NC
        F = localFace.shape[0] 
        FV = localFace.shape[1] 

        totalFace = self.total_face(localFace)
        index = np.sort(totalFace, axis=-1)
        I = index[:, 0]
        I += index[:, 1]*(index[:, 1] + 1)//2
        I += index[:, 2]*(index[:, 2] + 1)*(index[:, 2] + 2)//6
        if FV == 4: 
            I += index[:, 3]*(index[:, 3] + 1)*(index[:, 3] + 2)*(index[:, 3] + 3)//24
        _, i0, j = np.unique(I, return_index=True, return_inverse=True)

        NF = i0.shape[0]
        face = totalFace[i0, :]
        face2cell = np.zeros((NF, 4), dtype=self.itype)

        i1 = np.zeros(NF, dtype=self.itype)
        i1[j] = np.arange(F*NC, dtype=self.itype)

        face2cell[:, 0] = i0//F
        face2cell[:, 1] = i1//F
        face2cell[:, 2] = i0%F
        face2cell[:, 3] = i1%F

        return face, face2cell


class LagrangeWedgeMesh(Mesh3d):
    def __init__(self, node, cell, p=1, domain=None):

        self.p = p

        self.GD = node.shape[1]
        self.TD = 3
        self.ftype = node.dtype
        self.itype = cell.dtype
        self.meshtype = 'lwedge'

        self.domain = domain

        ds = LinearWedgeMeshDataStructure(node.shape[0], cell) # 线性网格的数据结构
        self.ds = LagrangeWedgeMeshDataStructure(ds, p)

        if self.p == 1:
            self.node = node
        else:
            NN = node.shape[0]
            self.node = np.zeros((NN, self.GD), dtype=self.ftype)
            bc0 = multi_index_matrix[2](self.p)/self.p
            bc1 = multi_index_matrix[1](self.p)/self.p
            bc = np.einsum('im, jn->ijmn', bc0, bc1).reshape(-1, 6)
            self.node = np.einsum('ijn, kj->ikn', node[cell], bc)

        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.multi_index_matrix = multi_index_matrix

    def entity(self, etype='cell'):
        if etype in {'cell', 3}:
            return self.ds.cell
        elif etype in {'face', 2}:
            return self.ds.tface, self.ds.qface
        elif etype in {'edge', 1}:
            return self.ds.edge
        elif etype in {'node', 0}:
            return self.node
        else:
            raise ValueError("`etype` is wrong!")

    def reference_cell_measure(self):
        return 0.5

    def number_of_corner_nodes(self):
        """
        Notes
        -----

        拉格朗日三角形网格中的节点分为单元角点节点, 边内部节节点和单元内部节点.

        这些节点默认的编号顺序也是: 角点节点, 边内部节点, 单元内部节点.

        该函数返回角点节点的个数.
        """
        return self.ds.NCN

    def integrator(self, k, etype='cell', ftype=None):
        qf0 = TriangleQuadrature(k)
        qf1 = GaussLegendreQuadrature(k)
        if etype in {'cell', 3}:
            return TensorProductQuadrature((qf0, qf1)) 
        elif etype in {'face', 2}:
            if ftype == 'tri':
                return qf0 
            elif ftype == 'quad':
                return TensorProductQuadrature((qf1, qf1)) 
            else:
                raise ValueError('the integrator `ftype` is not given! `tri` or `quad`'.format(face)) 
        elif etype in {'edge', 1}:
            return qf1 

    def entity_barycenter(self, etype=3, ftype=None, index=np.s_[:]):
        GD = self.geo_dimension()
        if etype in {'cell', 3}:
            qf = self.integrator(1, etype=3)
            bc, ws = qf.get_quadrature_points_and_weights()
            p = self.bc_to_point(bc, index=index).reshape(-1, GD)
        elif etype in {'face', 2}:
            if ftype == 'tri':
                qf = self.integrator(1, etype=2, ftype='tri')
                bcs, ws = qf.get_quadrature_points_and_weights()
                p = self.bc_to_point(bc, index=index).reshape(-1, GD)
            elif ftype == 'quad':
                qf = self.integrator(1, etype=2, ftype='quad')
                bcs, ws = qf.get_quadrature_points_and_weights()
                p = self.bc_to_point(bc, index=index).reshape(-1, GD)
            else:
                raise ValueError('the entity `ftype` is not given! `tri` or `quad`'.format(face)) 
        elif etype in {'edge', 1}:
            qf = self.integrator(1, etype=1)
            bc, ws = qf.get_quadrature_points_and_weights()
            p = self.bc_to_point(bc, index=index).reshape(-1, GD)
        elif etype in {'node', 0}:
            p = node[index]
        else:
            raise ValueError('the entity `{}` is not correct!'.format(entity)) 
        return p 

    def cell_volume(self, q=None, index=None):
        """
        
        Notes
        -----
        计算单元体积
        """
        p = self.p
        q = p if q is None else q

        qf = self.integrator(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        G = self.first_fundamental_form(bcs)
        l = np.sqrt(np.linalg.det(G))
        vol = 0.5*np.einsum('i, ij->j', ws, l)
        return vol

    def jacobi_matrix(self, bc, index=np.s_[:], return_grad=False):
        """
        Notes
        -----
        计算参考单元 （xi, eta, zeta) 到实际 Lagrange 三棱柱 (x) 之间映射的
        Jacobi 矩阵.

        """

        gphi = self.grad_shape_function(bc)
        node = self.entity('node')
        cell = self.entity('cell')
        J = np.einsum(
                'ijn, ...ijk->...ink', node[cell], gphi)
        shape = (-1, ) + J.shape[-3:]
        if return_grad is False:
            return J
        else:
            return J, gphi

    def bc_to_point(self, bc, index=np.s_[:], etype='cell'):
        node = self.entity('node')
        cell = self.entity('cell')
        phi = self.shape_function(bc)
        p = np.einsum('...jk, jkn->...jn', phi, node[cell])
        return p

    def shape_function(self, bc, p=None):
        p = self.p if p is None else p
        phi0 = lagrange_shape_function(bc[0], p[0])
        phi1 = lagrange_shape_function(bc[1], p[1])

        # i 是积分点
        # j 是单元
        # m 是基函数
        phi = np.einsum('im, kn->ikmn', phi0, phi1)
        shape = phi.shape[:-2] + (-1, )
        phi = phi.reshape(shape) # 展平自由度
        shape = (-1, 1) + phi.shape[-1:] # 增加一个单元轴，方便广播运算
        phi = phi.reshape(shape) # 展平积分点
        return phi 


    def grad_shape_function(self, bc, p=None, index=np.s_[:], variables='u'):
        """

        Notes
        -----
        计算单元形函数关于参考单元变量 u=(xi, eta, zeta) 或者实际变量 x 梯度.
        lambda_0 = 1 - xi
        lambda_1 = xi

        lambda_2 = 1 - eta - zeta
        lambda_3 = eta
        lambda_4 = zeta

        """
        p = self.p if p is None else p

        Dlambda0 = np.array([[-1], [1]], dtype=self.ftype)
        Dlambda1 = np.array([[-1, -1], [1, 0], [0, 1]], dtype=self.ftype)

        phi0 = lagrange_shape_function(bc[0], p)
        phi1 = lagrange_shape_function(bc[1], p)

        R0 = lagrange_grad_shape_function(bc[0], p)
        R1 = lagrange_grad_shape_function(bc[1], p)

        gphi0 = np.einsum('...ij, jn->...in', R0, Dlambda0) # (..., ldof, 1)
        gphi1 = np.einsum('...ij, jn->...in', R1, Dlambda1) # (..., ldof, 2)

        Gphi0 = np.einsum('imt, kn->ikmn', gphi0, phi1)
        Gphi1 = np.einsum('kn, imt->kinmt', phi0, gphi1)
        n = Gphi0.shape[0]*Gphi0.shape[1]
        shape = (n, (p+1)*(p+1)*(p+2)//2, 3)
        gphi = np.zeros(shape, dtype=self.ftype)
        gphi[..., 0].flat = Gphi0.flat
        gphi[..., 1:].flat = Gphi1.flat

        if variables == 'u':
            return gphi[..., None, :, :] #(..., 1, ldof, 3) 增加一个单元轴
        elif variables == 'x':
            G, J = self.first_fundamental_form(bc, index=index,
                    return_jacobi=True)
            G = np.linalg.inv(G)
            gphi = np.einsum('...ikm, ...imn, ...ln->...ilk', J, G, gphi)
            return gphi
    
    def first_fundamental_form(self, bc, index=np.s_[:], 
            return_jacobi=False, return_grad=False):
        """
        Notes
        -----
            计算拉格朗日网格在积分点处的第一基本形式。
        """

        TD = 3
        J = self.jacobi_matrix(bc, index=index,
                return_grad=return_grad)
        
        if return_grad:
            J, gphi = J

        shape = J.shape[0:-2] + (TD, TD)
        G = np.zeros(shape, dtype=self.ftype)
        for i in range(TD):
            G[..., i, i] = np.sum(J[..., i]**2, axis=-1)
            for j in range(i+1, TD):
                G[..., i, j] = np.sum(J[..., i]*J[..., j], axis=-1)
                G[..., j, i] = G[..., i, j]
        if (return_jacobi is False) & (return_grad is False):
            return G
        elif (return_jacobi is True) & (return_grad is False): 
            return G, J
        elif (return_jacobi is False) & (return_grad is True): 
            return G, gphi 
        else:
            return G, J, gphi

    def uniform_refine0(self, n=1, surface=None, returnim=False):
        """"
        uniform refine prism mesh.
        one cell -> 8
        """
        p = self.p+1
        ldof = (p+1)*(p+1)*(p+2)//2
        w1 = np.zeros((p+1, 2), dtype=np.int8)
        w1[:, 0] = np.arange(p, -1, -1)
        w1[:, 1] = w1[-1::-1, 0]
        w2 = self.multi_index_matrix[2](p)
        w3 = np.einsum('ij, km->ijkm', w1, w2)

        w = np.zeros((ldof, 6), dtype=np.int8)
        w[:, 0:3] = w3[:, 0, :, :].reshape(-1, 3)
        w[:, 3:] = w3[:, 1, :, :].reshape(-1, 3)

        for i in range(n):
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            node = self.entity('node')
            cell = self.entity('cell') # TODO: for high order 
            ps = np.einsum('im, km->ikm', cell + (NN + NC), w)
            ps.sort()
            _, i0, j = np.unique(
                    ps.reshape(-1, 6),
                    return_index=True,
                    return_inverse=True,
                    axis=0)
            ps = np.einsum('km, imd->ikd', w/p/p, node[cell]).reshape(-1, 3)
            self.node = ps[i0]

            cell2newNode = j.reshape(-1, ldof)
            cell = np.zeros((8*NC, 6), dtype=np.int)
            cell[0*NC:1*NC] = cell2newNode[:, [0, 1, 2, 6, 7, 8]]
            cell[1*NC:2*NC] = cell2newNode[:, [1, 3, 4, 7, 9, 10]]
            cell[2*NC:3*NC] = cell2newNode[:, [4, 2, 1, 10, 8, 7]]
            cell[3*NC:4*NC] = cell2newNode[:, [2, 4, 5, 8, 10, 11]]

            cell[4*NC:5*NC] = cell2newNode[:, [6, 7, 8, 12, 13, 14]]
            cell[5*NC:6*NC] = cell2newNode[:, [7, 9, 10, 13, 15, 16]]
            cell[6*NC:7*NC] = cell2newNode[:, [7, 10, 8, 13, 16, 14]]
            cell[7*NC:8*NC] = cell2newNode[:, [8, 10, 11, 14, 16, 17]]
            NN = len(i0)
            ds = LinearPrismMeshDataStructure(NN, cell)
            self.ds = LagrangePrismMeshDataStructure(ds, self.p)
    
    def uniform_refine1(self, n=1, surface=None, returnim=False):
        """"
        uniform refine prism mesh.
        one cell -> 2
        """
        p = self.p+1
        ldof = (p+1)*(self.p+1)*(self.p+2)//2
        w1 = np.zeros((p+1, 2), dtype=np.int8)
        w1[:, 0] = np.arange(p, -1, -1)
        w1[:, 1] = w1[-1::-1, 0]
        w2 = self.multi_index_matrix[2](self.p)
        w3 = np.einsum('ij, km->ijkm', w1, w2)

        w = np.zeros((ldof, 6), dtype=np.int8)
        w[:, 0:3] = w3[:, 0, :, :].reshape(-1, 3)
        w[:, 3:] = w3[:, 1, :, :].reshape(-1, 3)

        for i in range(n):
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            node = self.entity('node')
            cell = self.entity('cell') # TODO: for high order 
            ps = np.einsum('im, km->ikm', cell + (NN + NC), w)
            ps.sort()
            _, i0, j = np.unique(
                    ps.reshape(-1, 6),
                    return_index=True,
                    return_inverse=True,
                    axis=0)
            ps = np.einsum('km, imd->ikd', w/p/p, node[cell]).reshape(-1, 3)
            self.node = ps[i0]

            cell2newNode = j.reshape(-1, ldof)
            cell = np.zeros((2*NC, 6), dtype=np.int)
            cell[0*NC:1*NC] = cell2newNode[:, [0, 1, 2, 3, 4, 5]]
            cell[1*NC:2*NC] = cell2newNode[:, [3, 4, 5, 6, 7, 8]]
            NN = len(i0)
            
            ds = LinearPrismMeshDataStructure(NN, cell)
            self.ds = LagrangePrismMeshDataStructure(ds, self.p)
    
    def vtk_cell_type(self, etype='cell'):
        """

        Notes
        -----
            返回网格单元对应的 vtk 类型。
        """
        if etype in {'cell', 3}:
            VTK_LAGRANGE_WEDGE = 73
            return VTK_LAGRANGE_WEDGE
        elif etype in {'face', 2}:
            VTK_LAGRANGE_QUADRILATERAL = 70
            VTK_LAGRANGE_TRIANGLE = 69
            return VTK_LAGRANGE_TRIANGLE, VTK_LAGRANGE_QUADRILATERAL
        elif etype in {'edge', 1}:
            VTK_LAGRANGE_CURVE = 68
            return VTK_LAGRANGE_CURVE

    def to_vtk(self, etype='cell', index=np.s_[:], fname=None):
        """
        Parameters
        ----------

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        from .vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
#        idx = vtk_cell_index(self.p, cellType) # 转化为 vtk 编号顺序
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell]
        # 这里cell的编号顺序与vtk默认顺序一致
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)
    
class LagrangeWedgeMeshDataStructure(Mesh3dDataStructure):
    def __init__(self, ds, p):
        """

        Notes
        -----
        给定一个线性网格的数据结构，构造 p 次拉格朗日网格的数据结构
        """
        self.itype = ds.itype

        self.p = p
        self.V = (p + 1)*(p + 2)*(p + 1)//2
        self.E = ds.E 
        self.F = ds.F
        self.EV = ds.EV
        self.FV = ds.FV

        self.NCN = ds.NN  # 角点的个数
        self.NN = ds.NN 
        self.NE = ds.NE 
        self.NC = ds.NC 

        self.tface2cell = ds.tface2cell
        self.qface2cell = ds.qface2cell
        self.cell2edge = ds.cell2edge 


        if p == 1:
            self.cell = ds.cell
            self.edge = ds.edge
            self.tface = ds.tface
            self.qface = ds.qface
        else:
            NE = ds.NE
            edge = ds.edge
            self.edge = np.zeros((NE, p+1), dtype=self.itype)
            self.edge[:, [0, -1]] = edge
            flag = edge[:, 0] < edge[:, 1]
            idx0, = np.nonzero(flag)
            idx1, = np.nonzero(~flag)
            self.edge[idx0, 1:-1] = idx0[:, None]*(p-1) + self.NN + np.arange(ds.NN, ds.NN + p-1) 
            self.edge[idx1, -1:0:-1] = idx1[:, None]*(p-1) + np.arange(ds.NN, ds.NN + p-1)
            self.NN += NE*(p-1)

            # 三角形面
            NTF = ds.NTF
            self.tface = np.zeros((NTF, (p+1)*(p+2)//2), dtype=self.itype)
            self.tface[:, [0, -p-1, -1]] = ds.tface

            flag = ds.tface[:, 1] < ds.tface[:, 2]
            idx0, = np.nonzero(flag)
            idx1, = np.nonzero(~flag)
            self.tface[idx0, -p-1:-1] =  
            self.tface[idx1, -1:-p:-1] 

            self.tface[:, 
            self.NN += NTF*(p-2)*(p-1)//2

            # 四边形面
            NQF = ds.NQF
            self.qface = np.zeros((NQF, (p+1)*(p+1)), dtype=self.itype)
            self.qface[:, [0, p, -p-1, -1]] = ds.qface
            self.NN += NQF*(p-1)*(p-1)



            self.cell = np.zeros((self.NC, self.V), dtype=self.itype)
