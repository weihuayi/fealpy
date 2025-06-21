import numpy as np

from ...quadrature import TriangleQuadrature, GaussLegendreQuadrature

from .Mesh2d import Mesh2d, Mesh2dDataStructure

from .TriangleMesh import TriangleMesh

# 单纯形网格的多重指标矩阵
from .core import multi_index_matrix
# 单纯形网格的拉格朗日形函数
from .core import lagrange_shape_function 
# 单纯形网格拉格朗形函数关于重心坐标的导数
from .core import lagrange_grad_shape_function
from .core import LinearMeshDataStructure

class LinearTriangleMeshDataStructure(LinearMeshDataStructure):
    localEdge = np.array([(1, 2), (2, 0), (0, 1)])
    ccw = np.array([0, 1, 2])
    NVC = 3
    NEC = 3
    NVE = 2
    def __init__(self, NN, cell):
        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.itype = cell.dtype
        self.construct_edge()

class LagrangeTriangleMesh(Mesh2d):
    def __init__(self, node, cell, p=1, surface=None, boundary=None):

        self.p = p
        self.GD = node.shape[1]
        self.TD = 2
        self.meshtype = 'ltri'
        self.ftype = node.dtype
        self.itype = cell.dtype
        self.surface = surface
        self.boundary = boundary

        ds = LinearTriangleMeshDataStructure(node.shape[0], cell) 
        self.ds = LagrangeTriangleMeshDataStructure(ds, p)

        if p == 1:
            self.node = node
        else:
            NN = self.number_of_nodes()
            self.node = np.zeros((NN, self.GD), dtype=self.ftype)
            bc = multi_index_matrix[2](p)/p # (NQ, TD+1)
            self.node[self.ds.cell] = np.einsum('ijn, kj->ikn', node[cell], bc)

        if surface is not None:
            self.node, _ = surface.project(self.node)

        if boundary is not None:
            isBdNode = self.ds.boundary_node_flag()
            self.node[isBdNode], _ = self.boundary.project(self.node[isBdNode])

   
        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}
        self.multi_index_matrix = multi_index_matrix
    def geo_dimension(self):
        return self.GD

    def reference_cell_measure(self):
        return 0.5

    def number_of_corner_nodes(self):
        """
        Notes
        -----

        拉格朗日三角形网格中的节点分为单元角点节点，边内部节节点和单元内部节点。

        这些节点默认的编号顺序也是：角点节点，边内部节点，单元内部节点。

        该函数返回角点节点的个数。
        """
        return self.ds.NCN

    def lagrange_dof(self, p, spacetype='C'):
        if spacetype == 'C':
            return CLagrangeTriangleDof2d(self, p)
        elif spacetype == 'D':
            return DLagrangeTriangleDof2d(self, p)

    def vtk_cell_type(self, etype='cell'):
        """

        Notes
        -----
            返回网格单元对应的 vtk类型。
        """
        if etype in {'cell', 2}:
            VTK_LAGRANGE_TRIANGLE = 69
            return VTK_LAGRANGE_TRIANGLE 
        elif etype in {'face', 'edge', 1}:
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
        from ..vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        idx = vtk_cell_index(self.p, cellType)
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell[:, idx]]
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)

    def uniform_refine(self, n=1):
        p = self.p
        for i in range(n):
            NCN = self.number_of_corner_nodes()
            NC = self.number_of_cells()
            NE = self.number_of_edges()
            node = self.entity('node')
            cell = self.entity('cell')
            cell2edge = self.ds.cell_to_edge()
            cell2edge += NCN
            edgeCenter = self.entity_barycenter('edge')

            if self.surface is not None:
                edgeCenter, _ = self.surface.project(edgeCenter)

            if self.boundary is not None:
                isBdEdge = self.ds.boundary_edge_flag()
                edgeCenter[isBdEdge], _ = self.boundary.project(edgeCenter[isBdEdge])

            cp = [0, -p-1, -1]
            newCell = np.zeros((4*NC, 3), dtype=self.itype)
            newCell[0::4, 0] = cell[:, cp[0]] 
            newCell[0::4, 1] = cell2edge[:, 2] 
            newCell[0::4, 2] = cell2edge[:, 1]

            newCell[1::4, 0] = cell2edge[:, 2] 
            newCell[1::4, 1] = cell[:, cp[1]] 
            newCell[1::4, 2] = cell2edge[:, 0]

            newCell[2::4, 0] = cell2edge[:, 1] 
            newCell[2::4, 1] = cell2edge[:, 0] 
            newCell[2::4, 2] = cell[:, cp[2]]

            newCell[3::4, 0] = cell2edge[:, 0] 
            newCell[3::4, 1] = cell2edge[:, 1] 
            newCell[3::4, 2] = cell2edge[:, 2]

            node = np.r_['0', self.node[:NCN], edgeCenter]
            ds = LinearTriangleMeshDataStructure(node.shape[0], newCell) # 线性网格的数据结构
            self.ds = LagrangeTriangleMeshDataStructure(ds, p)

            if p == 1:
                self.node = node
            else:
                NN = self.number_of_nodes()
                self.node = np.zeros((NN, self.GD), dtype=self.ftype)
                bc = multi_index_matrix[2](p)/p
                # c: 单元指标
                # f: 面指标
                # e: 边指标
                # v: 顶点个数指标
                # i, j, k, d: 自由度或基函数指标
                # q: 积分点或重心坐标点指标
                # m, n: 空间或拓扑维数指标
                self.node[self.ds.cell] = np.einsum('cvn, iv->cin', node[newCell], bc)

                NCN = self.number_of_corner_nodes()
                if self.surface is not None:
                    self.node[NCN:], _ = self.surface.project(self.node[NCN:]) 

                if self.boundary is not None:
                    isBdNode = self.ds.boundary_node_flag()
                    self.node[isBdNode], _ = self.boundary.project(self.node[isBdNode])


    def integrator(self, q, etype='cell'):
        if etype in {'cell', 2}:
            return TriangleQuadrature(q)
        elif etype in {'edge', 'face', 1}:
            return GaussLegendreQuadrature(q)

    def cell_area(self, q=None, index=np.s_[:]):
        """

        Notes
        -----
        计算单元的面积。
        """
        p = self.p
        q = p if q is None else q
        GD = self.geo_dimension()

        qf = self.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = self.jacobi_matrix(bcs, index=index)
        n = np.cross(J[..., 0], J[..., 1], axis=-1)
        if GD == 3:
            n = np.sqrt(np.sum(n**2, axis=-1))
        a = np.einsum('i, ij->j', ws, n)/2.0
        return a

    def edge_length(self, q=None, index=np.s_[:]):
        """

        Note
        ----
        计算边的长度
        """
        p = self.p
        q = p if q is None else q

        qf = self.integrator(q, etype='edge')
        bcs, ws = qf.get_quadrature_points_and_weights() 

        J = self.jacobi_matrix(bcs, index=index)
        l = np.sqrt(np.sum(J**2, axis=(-1, -2)))
        a = np.einsum('i, ij->j', ws, l)
        return a

    def cell_unit_normal(self, bc, index=np.s_[:]):
        """

        Notes
        -----
        计算曲面情形下，积分点处的单位法线方向。
        """
        J = self.jacobi_matrix(bc, index=index)

        # n.shape 
        n = np.cross(J[..., 0], J[..., 1], axis=-1)
        if self.GD == 3:
            l = np.sqrt(np.sum(n**2, axis=-1, keepdims=True))
            n /= l

        return n

    def bc_to_point(self, bc, index=np.s_[:], etype='cell'):
        """

        Notes
        -----

        etype 这个参数实际上是不需要的，为了向后兼容，所以这里先保留。

        因为 bc 最后一个轴的长度就包含了这个信息。
        """
        node = self.node
        TD = bc.shape[-1] - 1
        entity = self.entity(etype=TD)[index] # 
        phi = self.shape_function(bc) # (NQ, 1, ldof)
        p = np.einsum('ijk, jkn->ijn', phi, node[entity])
        return p
    
    def jacobi_matrix(self, bc, index=np.s_[:], return_grad=False):
        """
        Notes
        -----
        计算参考单元 （xi, eta) 到实际 Lagrange 三角形(x) 之间映射的 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """

        TD = bc.shape[-1] - 1
        entity = self.entity(etype=TD)[index]
        gphi = self.grad_shape_function(bc, index=index)
        J = np.einsum(
                'cin, ...cim->...cnm',
                self.node[entity[index], :], gphi) #(NC,ldof,GD),(NQ,NC,ldof,TD)
        if return_grad is False:
            return J #(NQ,NC,GD,TD)
        else:
            return J, gphi

    def jacobi_TMOP(self, index = np.s_[:]):

        '''
        Notes
        -----
        计算参考单元 （xi, eta) 到实际 Lagrange 三角形(x) 之间映射的 Jacobi 矩阵
        分解出的各个度量
        '''

        p = self.p
        bc = multi_index_matrix[2](p)/p

        J = self.jacobi_matrix(bc, index=index)# Jacobi矩阵
        
        Lambda = np.sqrt(np.cross(J[..., 0], J[..., 1], axis = -1))# 网格单元尺寸
        
        r = np.sqrt(np.einsum('...ij->...j', J**2))# [r1,r2]

        Delta = np.einsum('ij,...j->...ij', np.eye(2), r)
        r0 = r[:, :, 0]*r[:, :, 1]
        Delta = np.einsum('...ijk,...i->...ijk',Delta, np.sqrt(1/r0))# 网格单元纵横比
        
        sphi = np.cross(J[..., 0], J[..., 1],axis=-1)/(r[...,0]*r[...,1])# sin(phi)
        cphi = np.sum(J[..., 0]*J[..., 1],axis=-1)/(r[...,0]*r[...,1])# cos(phi)

        E = np.zeros(J[...,1].shape)
        E[...,0] = 1
        stheta = np.cross(E,J[...,0],axis=-1)/r[...,0]
        ctheta = np.sum(J[...,0]*E,axis=-1)/r[...,0]
        
        V = np.zeros(J.shape)# 网格单元方向
        V[..., 0, 0] = ctheta
        V[..., 1, 0] = stheta
        V[..., 0, 1] = -stheta
        V[..., 1, 1] = ctheta
        
        Q = np.zeros(J.shape)# 网格单元夹角
        Q[..., 0, 0] = 1/np.sqrt(sphi)
        Q[..., 1, 0] = cphi/np.sqrt(sphi)
        Q[..., 1, 1] = sphi/np.sqrt(sphi)
        
        U = np.zeros(J.shape)# 网格单元尺寸和形状
        U[...,0,0] = r[..., 0]
        U[...,1,0] = r[..., 1]*cphi
        U[...,1,1] = r[..., 1]*sphi

        S = np.einsum('...ijk,...i->...ijk', U, 1/Lambda)# 网格单元形状
        #V = np.dot(J, np.linalg.inv(U))# 网格单元方向
        return J, Lambda, Q, Delta, S, U, V


    def first_fundamental_form(self, bc, index=np.s_[:], 
            return_jacobi=False, return_grad=False):
        """
        Notes
        -----
            计算网格曲面在积分点处的第一基本形式。
        """

        TD = bc.shape[-1] - 1
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

    def second_fundamental_form(self, bc, index=np.s_[:]):
        """
        Notes
        -----
            计算网格曲面在积分点处的第二基本形式。
        """
        pass

    def third_fundamental_form(self, bc, index=np.s_[:]):
        """
        Notes
        -----
            计算网格曲面在积分点处的第三基本形式。
        """
        pass


    def shape_function(self, bc, p=None):
        """

        Notes
        -----
            默认返回网格单元的形函数在积分点的值，否则返回空间基函数在积分点处的
            值
        """
        p = self.p if p is None else p
        phi = lagrange_shape_function(bc, p)
        return phi[..., None, :]


    def grad_shape_function(self, bc, p=None, index=np.s_[:], variables='u'):
        """

        Notes
        -----
        计算单元形函数关于参考单元变量 u=(xi, eta) 或者实际变量 x 梯度。

        lambda_0 = 1 - xi - eta
        lambda_1 = xi
        lambda_2 = eta

        """
        p = self.p if p is None else p 
        TD = bc.shape[-1] - 1
        if TD == 2:
            Dlambda = np.array([[-1, -1], [1, 0], [0, 1]], dtype=self.ftype)
        else:
            Dlambda = np.array([[-1], [1]], dtype=self.ftype)
        R = lagrange_grad_shape_function(bc, p) # (..., ldof, TD+1)
        gphi = np.einsum('...ij, jn->...in', R, Dlambda) # (..., ldof, TD)

        if variables == 'u':
            return gphi[..., None, :, :] #(..., 1, ldof, TD)
        elif variables == 'x':
            G, J = self.first_fundamental_form(bc, index=index, return_jacobi=True)
            G = np.linalg.inv(G)
            gphi = np.einsum('...ikm, ...imn, ...ln->...ilk', J, G, gphi) 
            return gphi
                    

    def print(self):

        """

        Notes
        -----
            打印网格信息， 用于调试。
        """

        node = self.entity('node')
        print('node:')
        for i, v in enumerate(node):
            print(i, ": ", v)

        edge = self.entity('edge')
        print('edge:')
        for i, e in enumerate(edge):
            print(i, ": ", e)

        cell = self.entity('cell')
        print('cell:')
        for i, c in enumerate(cell):
            print(i, ": ", c)

        edge2cell = self.ds.edge_to_cell()
        print('edge2cell:')
        for i, ec in enumerate(edge2cell):
            print(i, ": ", ec)


class LagrangeTriangleMeshDataStructure(Mesh2dDataStructure):
    def __init__(self, ds, p):
        """

        Notes
        -----
        给定一个线性网格的数据结构，构造 p 次拉格朗日网格的数据结构
        """
        self.itype = ds.itype

        self.NVC = (p+1)*(p+2)//2 # 单元顶点个数
        self.NEC = ds.NEC # 单元边的个数
        self.NVE = p+1 # 边上节点的个数

        self.NCN = ds.NN  # 角点的个数
        self.NN = ds.NN 
        self.NE = ds.NE 
        self.NC = ds.NC 

        self.ccw = np.zeros(3*p, dtype=self.itype)
        self.ccw[0:p] = np.cumsum(range(p))
        self.ccw[p:2*p] = range(self.NVC - p - 1, self.NVC-1)
        self.ccw[3*p-1:2*p-1:-1] = np.cumsum(range(2, p+2))  

        self.edge2cell = ds.edge2cell 

        if p == 1:
            self.cell = ds.cell
            self.edge = ds.edge
        else:
            NE = ds.NE
            edge = ds.edge
            self.edge = np.zeros((NE, p+1), dtype=self.itype)
            self.edge[:, [0, -1]] = edge
            self.edge[:, 1:-1] = self.NN + np.arange(NE*(p-1)).reshape(NE, p-1)
            self.NN += NE*(p-1)

            self.cell = np.zeros((self.NC, self.NVC), dtype=self.itype)

            index = multi_index_matrix[2](p)
            edge2cell = ds.edge2cell
            flag = edge2cell[:, 2] == 0
            self.cell[edge2cell[flag, 0][:, None], index[:, 0] == 0] = self.edge[flag]
            flag = edge2cell[:, 2] == 1
            self.cell[edge2cell[flag, 0][:, None], index[:, 1] == 0] = self.edge[flag, -1::-1]
            flag = edge2cell[:, 2] == 2
            self.cell[edge2cell[flag, 0][:, None], index[:, 2] == 0] = self.edge[flag]

            flag = (edge2cell[:, 3] == 0) & (edge2cell[:, 0] != edge2cell[:, 1])
            self.cell[edge2cell[flag, 1][:, None], index[:, 0] == 0] = self.edge[flag, -1::-1]
            flag = (edge2cell[:, 3] == 1) & (edge2cell[:, 0] != edge2cell[:, 1])
            self.cell[edge2cell[flag, 1][:, None], index[:, 1] == 0] = self.edge[flag]
            flag = (edge2cell[:, 3] == 2) & (edge2cell[:, 0] != edge2cell[:, 1])
            self.cell[edge2cell[flag, 1][:, None], index[:, 2] == 0] = self.edge[flag, -1::-1]

            if p > 2:
                flag = (index[:, 0] != 0) & (index[:, 1] != 0) & (index[:, 2] !=0)
                cdof = self.NVC - 3*p
                self.cell[:, flag]= self.NN + np.arange(self.NC*cdof).reshape(self.NC, cdof)
                self.NN += self.NC*cdof


class CLagrangeTriangleDof2d():
    """

    Notes
    -----
    拉格朗日三角形网格上的自由度管理类。
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = multi_index_matrix[2](p)
        self.itype = mesh.itype
        self.ftype = mesh.ftype

    def is_boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('edge', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        edge2dof = self.edge_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[edge2dof[index]] = True
        return isBdDof

    @property
    def face2dof(self):
        return self.edge_to_dof()
   
    def face_to_dof(self):
        return self.edge_to_dof()

    @property
    def edge2dof(self):
        return self.edge_to_dof()

    def edge_to_dof(self):
        """

        TODO
        ----
        1. 只取一部分边上的自由度
        """
        p = self.p
        mesh = self.mesh
        edge = mesh.entity('edge')

        if p == mesh.p:
            return edge

        NN = mesh.number_of_corner_nodes()
        NE = mesh.number_of_edges()
        edge2dof = np.zeros((NE, p+1), dtype=np.int_)
        edge2dof[:, [0, -1]] = edge[:, [0, -1]] # edge 可以是高次曲线
        if p > 1:
            edge2dof[:, 1:-1] = NN + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    @property
    def cell2dof(self):
        """
        
        Notes
        -----
            把这个方法属性化，保证程序接口兼容性
        """
        return self.cell_to_dof()


    def cell_to_dof(self):
        """

        TODO
        ----
        1. 只取一部分单元上的自由度。
        2. 用更高效的方式来生成单元自由度数组。
        """

        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell') # cell 可以是高次单元
        if p == mesh.p:
            return cell 

        # 空间自由度和网格的自由度不一致时，重新构造单元自由度矩阵
        NN = mesh.number_of_corner_nodes() # 注意这里只是单元角点的个数
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        edge2cell = mesh.ds.edge_to_cell()
        ldof = self.number_of_local_dofs()
        cell2dof = np.zeros((NC, ldof), dtype=np.int_)
        edge2dof = self.edge_to_dof()

        index = self.multiIndex
        flag = edge2cell[:, 2] == 0
        cell2dof[edge2cell[flag, 0][:, None], index[:, 0] == 0] = edge2dof[flag]
        flag = edge2cell[:, 2] == 1
        cell2dof[edge2cell[flag, 0][:, None], index[:, 1] == 0] = edge2dof[flag, -1::-1]
        flag = edge2cell[:, 2] == 2
        cell2dof[edge2cell[flag, 0][:, None], index[:, 2] == 0] = edge2dof[flag]

        flag = (edge2cell[:, 3] == 0) & (edge2cell[:, 0] != edge2cell[:, 1])
        cell2dof[edge2cell[flag, 1][:, None], index[:, 0] == 0] = edge2dof[flag, -1::-1]
        flag = (edge2cell[:, 3] == 1) & (edge2cell[:, 0] != edge2cell[:, 1])
        cell2dof[edge2cell[flag, 1][:, None], index[:, 1] == 0] = edge2dof[flag]
        flag = (edge2cell[:, 3] == 2) & (edge2cell[:, 0] != edge2cell[:, 1])
        cell2dof[edge2cell[flag, 1][:, None], index[:, 2] == 0] = edge2dof[flag, -1::-1]

        if p > 2:
            flag = (index[:, 0] != 0) & (index[:, 1] != 0) & (index[:, 2] !=0)
            cdof =  ldof - 3*p
            cell2dof[:, flag]= NN + NE*(p-1) + np.arange(NC*cdof).reshape(NC, cdof)

        return cell2dof

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        if p == mesh.p:
            return node

        cell2dof = self.cell_to_dof()
        GD = mesh.geo_dimension()
        gdof = self.number_of_global_dofs()
        ipoint = np.zeros((gdof, GD), dtype=np.float64)
        bcs = self.multiIndex/p # 计算插值点对应的重心坐标
        ipoint[cell2dof] = mesh.bc_to_point(bcs).swapaxes(0, 1) 
        return ipoint

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh

        if p == mesh.p:
            return mesh.number_of_nodes()

        gdof = mesh.number_of_corner_nodes() # 注意这里只是单元角点的个数
        if p > 1:
            NE = self.mesh.number_of_edges()
            gdof += (p-1)*NE

        if p > 2:
            ldof = self.number_of_local_dofs()
            NC = self.mesh.number_of_cells()
            gdof += (ldof - 3*p)*NC
        return gdof

    def number_of_local_dofs(self, doftype='cell'):
        p = self.p
        if doftype in {'cell', 2}:
            return (p+1)*(p+2)//2 
        elif doftype in {'face', 'edge',  1}:
            return self.p + 1
        elif doftype in {'node', 0}:
            return 1

class DLagrangeTriangleDof2d():
    """

    Notes
    -----
    拉格朗日四边形网格上的自由度管理类。
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = multi_index_matrix[2](p)
        self.itype = mesh.itype
        self.ftype = mesh.ftype

    @property
    def face2dof(self):
        return None 

    def face_to_dof(self):
        return None 

    @property
    def edge2dof(self):
        return None 

    def edge_to_dof(self):
        """

        TODO
        ----
        1. 只取一部分边上的自由度
        """
        return None

    @property
    def cell2dof(self):
        """
        
        Notes
        -----
            把这个方法属性化，保证程序接口兼容性
        """
        return self.cell_to_dof()


    def cell_to_dof(self):
        """

        TODO
        ----
        1. 只取一部分单元上的自由度。
        2. 用更高效的方式来生成单元自由度数组。
        """

        p = self.p
        NC = self.mesh.number_of_cells()
        ldof = self.number_of_local_dofs(doftype='cell')
        cell2dof = np.arange(NC*ldof).reshape(NC, ldof)
        return cell2dof

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        GD = self.mesh.geo_dimension()
        bcs = self.multiIndex/p # 计算插值点对应的重心坐标
        ipoint = mesh.bc_to_point(bcs).swapaxes(0, 1).reshape(-1, GD)
        return ipoint

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs(doftype='cell')
        return NC*ldof

    def number_of_local_dofs(self, doftype='cell'):
        p = self.p
        if doftype in {'cell', 2}:
            return (p+1)*(p+2)//2
        elif doftype in {'face', 'edge',  1}:
            return p + 1
        elif doftype in {'node', 0}:
            return 1
