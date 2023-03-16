import numpy as np

from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature

from .Mesh3d import Mesh3d, Mesh3dDataStructure
from .TetrahedronMesh import TetrahedronMesh

from .core import multi_index_matrix
from .core import lagrange_shape_function 
from .core import lagrange_grad_shape_function
from .core import LinearMeshDataStructure

class LinearHexahedronMeshDataStructure(LinearMeshDataStructure):
    # The following local data structure should be class properties
    localFace = np.array([
        (0, 2, 6, 4), (1, 5, 7, 3), # bottom and top faces
        (0, 1, 3, 2), (4, 6, 7, 5), # left and right faces  
        (0, 4, 5, 1), (6, 2, 3, 7)])# front and back faces
    localEdge = np.array([
        (0, 1), (0, 2), (0, 4), (1, 3),
        (1, 5), (2, 3), (2, 6), (3, 7),
        (4, 5), (4, 6), (5, 7), (6, 7)])
    localFace2edge = np.array([
        (1,  6, 9, 2), (4, 10, 7, 3),   # bottom and top faces
        (0,  3, 5, 1), (9, 11, 10, 8),  # left and right faces 
        (2,  8, 4, 0), (6, 5, 7,  11)]) # front and back faces

    NVE = 2 # 每个边有 2 个顶点
    NVF = 4 # 每个单元面有 4 个顶点 
    NVC = 8 # 每个单元 8 个顶点  

    NEF = 4 # 每个单元面有 4 条边
    NEC = 12 # 每个单元 12 条边 

    NFC = 6 # 每个单元  6 个面 

    def __init__(self, NN, cell):
        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.itype = cell.dtype
        self.construct_edge()
        self.construct_face()

class LagrangeHexahedronMesh(Mesh3d):
    def __init__(self, node, cell, p=1, domain=None):

        self.p = p
        self.GD = node.shape[1]
        self.TD = 2
        self.ftype = node.dtype
        self.itype = cell.dtype
        self.meshtype = 'lhex'

        self.domain = domain

        ds = LinearHexahedronMeshDataStructure(node.shape[0], cell) # 线性网格的数据结构
        self.ds = LagrangeHexahedronMeshDataStructure(ds, p)

        if p == 1:
            self.node = node
        else:
            pass

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {}
        self.celldata = {}
        self.meshdata = {}
        self.multi_index_matrix = multi_index_matrix


    def reference_cell_measure(self):
        return 1

    def number_of_corner_nodes(self):
        """
        Notes
        -----
        该函数返回角点节点的个数。
        """
        return self.ds.NCN

    def lagrange_dof(self, p, spacetype='C'):
        if spacetype == 'C':
            return CLagrangeHexahedronMeshDof(self, p)
        elif spacetype == 'D':
            return DLagrangeHexahedronDof(self, p)

    def vtk_cell_type(self, etype='cell'):
        """

        Notes
        -----
            返回网格单元对应的 vtk 类型。
        """
        if etype in {'cell', 3}:
            VTK_LAGRANGE_HEXAHEDRON = 72
            return VTK_LAGRANGE_HEXAHEDRON
        elif etype in {'face', 2}:
            VTK_LAGRANGE_QUADRILATERAL = 70 
            return VTK_LAGRANGE_QUADRILATERAL
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
        idx = vtk_cell_index(self.p, cellType) # 转化为 vtk 编号顺序
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

    def to_tetmesh(self):
        cell = self.entity('cell')
        node = self.entity('node')
        hexCell2face = self.ds.cell_to_face()
        localCell = np.array([
            [0, 6, 2, 7],
            [0, 2, 3, 7],
            [0, 3, 1, 7],
            [0, 1, 5, 7],
            [0, 5, 4, 7],
            [0, 4, 6, 7]], dtype=np.int_)
        cell = cell[:, localCell].reshape(-1, 4)
        data = self.meshdata
        celldata = self.celldata
        nodedata = self.nodedata

        mesh = TetrahedronMesh(node, cell)
        for key in celldata:
            mesh.celldata[key] = np.tile(celldata[key], (6, 1)).T.reshape(-1)
        mesh.meshdata = data
        mesh.nodedata = nodedata
        return mesh

    def shape_function(self, bc, p=None):
        """

        Notes
        -----

        bc 是一个长度为 TD 的 tuple

        bc[i] 是一个一维积分公式的重心坐标数组

        这里假设 bc[0] == bc[1] == ... = bc[TD-1] # 考虑积分点不同的情况

        """
        p = self.p if p is None else p
        TD = len(bc)
        phi = lagrange_shape_function(bc[0], p) # 线上的形函数
        if TD == 2: # 面上的形函数
            phi = np.einsum('im, jn->ijmn', phi, phi)
            shape = phi.shape[:-2] + (-1, )
            phi = phi.reshape(shape) # 展平自由度
            shape = (-1, 1) + phi.shape[-1:] # 增加一个单元轴，方便广播运算
            phi = phi.reshape(shape) # 展平积分点
        elif TD == 3: # 体上的形函数
            phi = np.einsum('il, jm, kn->ijklmn', phi, phi, phi)
            shape = phi.shape[:-3] + (-1, )
            phi = phi.reshape(shape) # 展平自由度
            shape = (-1, 1) + phi.shape[-1:] # 增加一个单元轴，方便广播运算
            phi = phi.reshape(shape) # 展平积分点
        return phi 

    def bc_to_point(self, bc, index=np.s_[:], etype='cell'):
        """

        Notes
        -----
        etype 这个参数实际上是不需要的，为了兼容，所以这里先保留。

        bc 是一个 tuple 数组

        Examples
        --------
        >>> bc = TensorProductQuadrature(3, TD=3) # 第三个张量积分公式
        >>> points = mesh.bc_to_point(bc)

        """
        node = self.node
        TD = len(bc) 
        entity = self.entity(etype=TD)[index] #
        phi = self.shape_function(bc) # (NQ, 1, ldof)
        p = np.einsum('...jk, jkn->...jn', phi, node[entity])
        return p

class LagrangeHexahedronMeshDataStructure(Mesh3dDataStructure):
    def __init__(self, ds, p):

        self.itype = ds.itype

        self.p = p
        self.NVC = (p+1)*(p+1)*(p+1) 
        self.NVE = p+1 # 每条边上有 p+1 个点
        self.NVF = (p+1)*(p+1)
        self.NEC = ds.NEC
        self.NFC = ds.NFC

        self.NCN = ds.NN  # 角点的个数
        self.NN = ds.NN 
        self.NE = ds.NE 
        self.NF = ds.NF
        self.NC = ds.NC 

        self.face2cell = ds.face2cell 
        self.cell2edge = ds.cell2edge

        if p == 1:
            self.cell = ds.cell
            self.face = ds.face
            self.edge = ds.edge
        else:
            pass


class CLagrangeHexahedronMeshDof():
    """

    Notes
    -----
    拉格朗日六面体网格上的连续空间的自由度管理类.
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.itype = mesh.itype
        self.ftype = mesh.ftype

    def is_boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        face2dof = self.face_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[face2dof[index]] = True
        return isBdDof

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
        else:
            pass

    @property
    def face2dof(self):
        return self.face_to_dof()

    def face_to_dof(self):
        """

        TODO
        ----
        1. 只取一部分面上的自由度
        """
        p = self.p
        mesh = self.mesh
        face = mesh.entity('face')
        if p == mesh.p:
            return face 
        else:
            pass


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
        else:
            pass


    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        if p == mesh.p:
            return node
        else:
            pass

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh

        if p == mesh.p:
            return mesh.number_of_nodes()
        else:
            pass

    def number_of_local_dofs(self, doftype='cell'):
        p = self.p
        if doftype in {'cell', 3}:
            return (p + 1)**3
        elif doftype in {'face', 2}:
            return (p + 1)**2
        elif doftype in {'edge', 1}:
            return (p + 1)
        elif doftype in {'node', 0}:
            return 1

class DLagrangeHexahedronDof():
    """

    Notes
    -----
        拉格朗日六面体网格上的间断自由度管理类。
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
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
        cell2dof = np.arange(NC*(p+1)**3).reshape(NC, (p+1)**3)
        return cell2dof

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')

        NC = mesh.number_of_cells()
        GD = mesh.geo_dimension()
        bc = multi_index_matrix[1](p)/p
        ipoint = mesh.bc_to_point((bc, bc, bc)).reshape(-1, NC,
                GD).swapaxes(0, 1).reshape(-1, GD)
        return ipoint

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()
        return NC*(p+1)*(p+1)*(p+1)

    def number_of_local_dofs(self, doftype='cell'):
        p = self.p
        if doftype in {'cell', 3}:
            return (p + 1)**3
        elif doftype in {'face', 2}:
            return (p + 1)**2
        elif doftype in {'edge', 1}:
            return (p + 1)
        elif doftype in {'node', 0}:
            return 1
