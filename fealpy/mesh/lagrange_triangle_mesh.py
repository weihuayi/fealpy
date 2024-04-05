import numpy as np

from .lagrange_mesh import LagrangeMesh
from .triangle_mesh import TriangleMesh
from .mesh_base import Mesh, Plotable

class LagrangeTriangleMeshDataStructure():
    def __init__(self, NN, cell):
        self.NN = NN
        self.TD = 2
        self.cell = cell

    def number_of_cells(self):
        return self.cell.shape[0]

class LagrangeTriangleMesh(LagrangeMesh): 
    def __init__(self, node, cell, surface=None, p=1):

        mesh = TriangleMesh(node, cell)
        NN = mesh.number_of_nodes()

        self.ftype = node.dtype
        self.itype = cell.dtype
        self.meshtype = 'ltri'

        self.GD = node.shape[1]

        self.p = p
        self.surface = surface

        self.node = mesh.interpolation_points(p)
        self.multi_index_matrix = mesh.multi_index_matrix

        if surface is not None:
            self.node, _ = surface.project(self.node)
        cell = mesh.cell_to_ipoint(p)

        NN = self.node.shape[0]
        self.ds = LagrangeTriangleMeshDataStructure(NN, cell)
        self.ds.edge = mesh.edge_to_ipoint(p=p)

        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}
        
        self.edge_bc_to_point = self.bc_to_point
        self.cell_bc_to_point = self.bc_to_point
        self.face_to_ipoint = self.edge_to_ipoint

        #self.shape_function = self._shape_function
        self.cell_shape_function = self._shape_function
        self.face_shape_function = self._shape_function
        self.edge_shape_function = self._shape_function

    def geo_dimension(self):
        return self.GD

    def ref_cell_measure(self):
        return 0.5
 
    def lagrange_dof(self, p, spacetype='C'):
        if spacetype == 'C':
            return CLagrangeTriangleDof2d(self, p)
        elif spacetype == 'D':
            return DLagrangeTriangleDof2d(self, p)
    
    def uniform_refine(self, n=1):
        p =self.p
        pass

    def integrator(self, q, etype='cell'):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        if etype in {'cell', 2}:
            from ..quadrature import TriangleQuadrature
            return TriangleQuadrature(q)
        elif etype in {'edge', 'face', 1}:
            from ..quadrature import GaussLegendreQuadrature
            return GaussLegendreQuadrature(q)
    
    def cell_area(self, q=None, index=np.s_[:]):
        """
        @berif 计算单元的面积。
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
        @berif 计算边的长度
        """
        p = self.p
        q = p if q is None else q

        qf = self.integrator(q, etype='edge')
        bcs, ws = qf.get_quadrature_points_and_weights() 

        J = self.jacobi_matrix(bcs, index=index)
        l = np.sqrt(np.sum(J**2, axis=(-1, -2)))
        a = np.einsum('i, ij->j', ws, l)
        return a


    def entity_measure(self, etype=2, index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 'face', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return 0
        else:
            raise ValueError(f"Invalid entity type '{etype}'.")
 
    def lagrange_shape_function(self, bc, p, n=0):
        """

        @berif 计算形状为 (..., TD+1) 的重心坐标数组 bc 中的每一个重心坐标处的 p 次
        Lagrange 形函数关于 TD+1 个重心坐标的 n 阶导数.
        
        注意当 n = 0 时, 返回的是函数值。
        """
        assert n <= p

        TD = bc.shape[-1] - 1
        multiIndex = self.multi_index_matrix(p, etype=TD) 
        ldof = multiIndex.shape[0] # p 次 Lagrange 形函数的个数 

        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1) # (NQ, p+1, TD+1)
        A = np.ones(shape, dtype=bc.dtype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)

        if n == 0:
            A[..., 1:, :] *= P.reshape(-1, 1)
            idx = np.arange(TD+1)
            R = np.prod(A[..., multiIndex, idx], axis=-1)
            return R 
        else:
            T = p*bc[..., None, :] - t.reshape(-1, 1) # (NQ, p, TD+1)
            F0 = A.copy() # (NQ, p+1, TD+1) 
            F1 = np.zeros(A.shape, dtype=bc.dtype)

            # (NQ, p, TD+1) = (NQ, p, TD+1)*(NQ, p, TD+1) + (NQ, p, TD+1)
            for i in range(1, n+1):
                for j in range(1, p+1):
                    F1[..., j, :] = F1[..., j-1, :]*T[..., j-1, :] + i*p*F0[..., j-1, :]
                F0[:] = F1

            A[..., 1:, :] *= P.reshape(-1, 1)
            F0[..., 1:, :] *= P.reshape(-1, 1)
            
            Q = A[..., multiIndex, range(TD+1)]
            M = F0[..., multiIndex, range(TD+1)]

            shape = bc.shape[:-1]+(ldof, TD+1) # (NQ, ldof, TD+1)
            R = np.zeros(shape, dtype=bc.dtype)
            for i in range(TD+1):
                idx = list(range(TD+1))
                idx.remove(i)
                R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)
            return R # (..., ldof, TD+1)

    def lagrange_grad_shape_function(self, bc, p): 
        """
        @berif 计算形状为 (..., TD+1) 的重心坐标数组 bc 中, 每一个重心坐标处的 p 次
        Lagrange 形函数值关于该重心坐标的梯度。
        """

        TD = bc.shape[-1] - 1
        multiIndex = self.multi_index_matrix(p, etype=TD) 
        ldof = multiIndex.shape[0] # p 次 Lagrange 形函数的个数

        c = np.arange(1, p+1)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=bc.dtype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=bc.dtype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]

        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=bc.dtype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)
        return R # (..., ldof, TD+1)

    def shape_function(self, bc, p=None):
        """
        @brief 网格空间基函数
        """
        p = self.p if p is None else p
        phi = self.lagrange_shape_function(bc, p)
        return phi[..., None, :]

    def grad_shape_function(self, bc, p=None, index=np.s_[:], variables='u'):
        """
        @berif 计算单元形函数关于参考单元变量 u=(xi, eta) 或者实际变量 x 梯度。

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
        R = self.lagrange_grad_shape_function(bc, p=p)  # (..., ldof, TD+1)
        gphi = np.einsum('...ij, jn->...in', R, Dlambda) # (..., ldof, TD)

        if variables == 'u':
            return gphi[..., None, :, :] #(..., 1, ldof, TD)
        elif variables == 'x':
            G, J = self.first_fundamental_form(bc, index=index, return_jacobi=True)
            G = np.linalg.inv(G)
            gphi = np.einsum('...ikm, ...imn, ...ln->...ilk', J, G, gphi) 
            return gphi

    def grad_shape_function_on_edge(self, bc, cindex, lidx, p=1, direction=True):
        """
        @brief 计算单元边上所有形函数在边上的积分点处的导函数值
        @param bc 边上的一组积分点
        @param cindex 边所在的单元编号
        @param lidx 边在该单元的局部编号
        @param direction Ture 表示边的方向和单元的逆时针方向一致，False 表示不一致
        """
        NC = len(cindex)
        nmap = np.array([1, 2, 0])
        pmap = np.array([2, 0, 1])
        shape = (NC, ) + bc.shape[0:-1] + (3, )
        bcs = np.zeros(shape, dtype=self.ftype)  # (NE, 3) or (NE, NQ, 3)
        idx = np.arange(NC)
        if direction:
            bcs[idx, ..., nmap[lidx]] = bc[..., 0]
            bcs[idx, ..., pmap[lidx]] = bc[..., 1]
        else:
            bcs[idx, ..., nmap[lidx]] = bc[..., 1]
            bcs[idx, ..., pmap[lidx]] = bc[..., 0]

        gphi = self.grad_shape_function(bcs, p=p, index=cindex, variables='x')

        return gphi

    grad_shape_function_on_face = grad_shape_function_on_edge
        

    def number_of_local_ipoints(self, p, iptype='cell'):
        """
        @brief
        """
        if iptype in {'cell', 2}:
            return (p+1)*(p+2)//2
        elif iptype in {'face', 'edge',  1}: # 包括两个顶点
            return p + 1
        elif iptype in {'node', 0}:
            return 1

    def number_of_global_ipoints(self, p):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        return NN + (p-1)*NE + (p-2)*(p-1)//2*NC

    def interpolation_points(self, p: int, index=np.s_[:]):
        """
        @brief 获取三角形网格上所有 p 次插值点
        """
        cell = self.entity('cell')
        node = self.entity('node')
        if p == 1:
            return node

        if p > 1:
            NN = self.number_of_nodes()
            GD = self.geo_dimension()

            gdof = self.number_of_global_ipoints(p)
            ipoints = np.zeros((gdof, GD), dtype=self.ftype)
            ipoints[:NN, :] = node

            NE = self.number_of_edges()

            edge = self.entity('edge')

            w = np.zeros((p-1, 2), dtype=np.float64)
            w[:, 0] = np.arange(p-1, 0, -1)/p
            w[:, 1] = w[-1::-1, 0]
            ipoints[NN:NN+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', w,
                    node[edge,:]).reshape(-1, GD)
        if p > 2:
            TD = self.top_dimension()
            multiIndex = self.multi_index_matrix(p, TD)
            isEdgeIPoints = (multiIndex == 0)
            isInCellIPoints = ~(isEdgeIPoints[:,0] | isEdgeIPoints[:,1] |
                    isEdgeIPoints[:,2])
            w = multiIndex[isInCellIPoints, :]/p
            ipoints[NN+(p-1)*NE:, :] = np.einsum('ij, kj...->ki...', w,
                    node[cell, :]).reshape(-1, GD)
        return ipoints # (gdof, GD)

    def cell_to_ipoint(self, p, index=np.s_[:]):
        """
        @berif 
        """
        pass
 
    def rot_lambda(self, index=np.s_[:]):
        """
        @berif 
        """
        pass
    def uniform_refine(self, n=1, surface=None, interface=None, returnim=False):
        """
        @berif 一致加密三角形网格
        """
        pass

    def vtk_cell_type(self, etype='cell'):
        """
        @berif  返回网格单元对应的 vtk类型。
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

        @berif 把网格转化为 VTK 的格式
        """
        from .vtk_extent import vtk_cell_index, write_to_vtu

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

class CLagrangeTriangleDof2d():
    """
    @berif 拉格朗日三角形网格上的自由度管理类。
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix(p, etype=2)
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
        edge2dof = np.zeros((NE, p+1), dtype=np.int)
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
    @berif 拉格朗日四边形网格上的自由度管理类。
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
