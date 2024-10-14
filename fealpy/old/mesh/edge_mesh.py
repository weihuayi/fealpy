import numpy as np

from scipy.sparse import csr_matrix
from numpy.typing import NDArray

from typing import Union
from types import ModuleType

from .mesh_base import Mesh, Plotable
from .mesh_data_structure import Mesh1dDataStructure, HomogeneousMeshDS

class EdgeMeshDataStructure(Mesh1dDataStructure, HomogeneousMeshDS):
    def __init__(self, NN, cell):
        self.NN = NN
        self.cell = cell 
        self.NC = len(cell)

    def construct(self) -> None: 
        """
        @brief 覆盖基类的 construct 函数
        """
        return None

    def node_to_cell(self):
        NN = self.NN
        NC = self.NC
        I = self.cell.flat
        J = np.repeat(range(NC), 2)
        val = np.ones(2*NC, dtype=np.bool_)
        node2edge = csr_matrix((val, (I, J)), shape=(NN, NC))
        return node2edge

    face_to_cell = node_to_cell

## @defgroup MeshGenerators Meshgeneration algorithms on commonly used domain 
## @defgroup MeshQuality
class EdgeMesh(Mesh, Plotable):
    def __init__(self, node, cell):
        self.node = node
        self.itype = cell.dtype
        self.ftype = node.dtype

        self.meshtype = 'edge'
        
        NN = len(node)
        self.ds = EdgeMeshDataStructure(NN, cell)

        self.nodedata = {}
        self.celldata = {}
        self.edgedata = self.celldata
        self.facedata = self.nodedata
        self.meshdata = {}

        self.cell_length = self.edge_length
        self.cell_tangent = self.edge_tangent
        self.cell_unit_tangent = self.edge_unit_tangent

        self.cell_to_ipoint = self.edge_to_ipoint
        self.face_to_ipoint = self.node_to_ipoint
        self.shape_function = self._shape_function

    def ref_cell_measure(self):
        return 1.0

    def ref_face_measure(self):
        return 0.0

    def integrator(self, q: int, etype: Union[str, int]='cell'):
        """
        @brief 返回第 k 个高斯积分公式。
        """
        from ..quadrature import GaussLegendreQuadrature
        return GaussLegendreQuadrature(q)

    def grad_shape_function(self, bc: NDArray, p: int=1, variables: str='x', index=np.s_[:]):
        """
        @brief 
        """
        R = self._grad_shape_function(bc, p=p)
        if variables == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = np.einsum('...ij, cjm->...cim', R, Dlambda)
            return gphi 
        else:
            return R

    def entity_measure(self, etype: Union[int, str]='cell', index=np.s_[:], node=None):
        """
        """
        if etype in {1, 'cell', 'edge'}:
            return self.cell_length(index=index, node=None)
        elif etype in {0, 'face', 'node'}:
            return np.array([0.0], dtype=self.ftype)
        else:
            raise ValueError(f"entity type: {etype} is wrong!")
    
    def grad_lambda(self, index=np.s_[:]):
        """
        @brief 计算所有单元上重心坐标函数的导数
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        v = node[cell[:, 1]] - node[cell[:, 0]]
        NC = len(cell) 
        GD = self.geo_dimension()
        Dlambda = np.zeros((NC, 2, GD), dtype=self.ftype)
        h2 = np.sum(v**2, axis=-1)
        v /=h2.reshape(-1, 1)
        Dlambda[:, 0, :] = -v
        Dlambda[:, 1, :] = v
        return Dlambda
    

    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        return p + 1

    def number_of_global_ipoints(self, p: int) -> int:
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        return NN + (p-1)*NC

    def interpolation_points(self, p: int, index=np.s_[:]) -> NDArray:
        GD = self.geo_dimension()
        node = self.entity('node')

        if p == 1:
            return node
        else:
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            gdof = NN + NC*(p-1)
            ipoint = np.zeros((gdof, GD), dtype=self.ftype)
            ipoint[:NN] = node
            cell = self.entity('cell')
            w = np.zeros((p-1,2), dtype=np.float64)
            w[:,0] = np.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            GD = self.geo_dimension()
            ipoint[NN:NN+(p-1)*NC] = np.einsum('ij, kj...->ki...', w,
                    node[cell]).reshape(-1, GD)

            return ipoint

    def face_unit_normal(self, index=np.s_[:], node=None):
        """
        @brief
        """
        raise NotImplementedError

    def cell_normal(self, index=np.s_[:], node=None):
        """
        @brief 单元的法线方向
        """
        assert self.geo_dimension() == 2
        v = self.cell_tangent(index=index, node=node)
        w = np.array([(0, -1),(1, 0)])
        return v@w

    @classmethod
    def from_inp_file(cls, fname):
        from .inp_file_reader import InpFileReader

        reader = InpFileReader(fname)
        reader.parse()

        # 提取所需的数据
        parts = reader.parts

        # 从 parts 中获取 node 和 element 数据
        part_name = next(iter(parts)) # parts 的名称，需要根据 .inp 文件来选择正确的 parts 名称
        part_data = parts[part_name]
        node_data = part_data['node']
        node = node_data[0]
        node_index = node_data[-1]
        element_data = part_data['elem']
        element_1 = element_data[list(element_data.keys())[0]]
        element_2 = element_data[list(element_data.keys())[-1]]
        cell = np.concatenate([data[0] for _, data in element_data.items()])
        mesh = cls(node, cell)

        nset_data = part_data['nset']

        elset_data = part_data['elset']

        orientation_data = part_data['orientation']

        solid_section_data = part_data['solid_section']

        beam_section_data = part_data['beam_section']

        assemblys = reader.assembly

        assembly_name = next(iter(assemblys)) # parts 的名称，需要根据 .inp 文件来选择正确的 parts 名称

        assembly_data = assemblys[assembly_name]

        instance_data = assembly_data['instance']

        nset_assembly_data = assembly_data['nset']

        return mesh

    ## @ingroup MeshGenerators
    @classmethod
    def from_triangle_mesh(cls, mesh):
        pass

    ## @ingroup MeshGenerators
    @classmethod
    def from_tetrahedron_mesh(cls, mesh):
        pass

    ## @ingroup MeshGenerators
    @classmethod
    def from_tower(cls):
        node = np.array([
            [-950, 0, 5080], [950, 0, 5080], [-950, 950, 2540], 
            [950, 950, 2540], [950, -950, 2540], [-950, -950, 2540],
            [-2540, 2540, 0], [2540, 2540, 0], [2540, -2540, 0], 
            [-2540, -2540, 0]], dtype=np.float64)
        cell = np.array([
            [0, 1], [3, 0], [1, 2], [1, 5], [0, 4], 
            [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
            [4, 3], [2, 3], [4, 5], [2, 9], [6, 5], 
            [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
            [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=np.int_)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (np.array([6, 7, 8, 9], dtype=np.int_), np.zeros(3))
        mesh.meshdata['force_bc'] = (np.array([0, 1], dtype=np.int_), np.array([0, 900, 0]))

        return mesh 

    ## @ingroup MeshGenerators
    @classmethod
    def from_four_bar_mesh(cls):
        # 单位为 mm
        node = np.array([
            [0, 0], [400, 0], 
            [400, 300], [0, 300]], dtype=np.float64)
        cell = np.array([
            [0, 1], [2, 1], 
            [0, 2], [3, 2]], dtype=np.int_)
        mesh = cls(node, cell)

        # 按分量处理自由度索引
        mesh.meshdata['disp_bc'] = (np.array([0, 1, 3, 6, 7], dtype=np.int_), np.zeros(1))
        mesh.meshdata['force_bc'] = (np.array([1, 2], dtype=np.int_), 
                                     np.array([[2e4, 0], [0, -2.5e4]], dtype=np.float64))

        return mesh 

    ## @ingroup MeshGenerators
    @classmethod
    def generate_balcony_truss_mesh(cls):
        # 单位为英寸 in
        node = np.array([
            [0, 0], [36, 0], 
            [0, 36], [36, 36], [72, 36]], dtype=np.float64)
        cell = np.array([
            [0, 1], [1, 2], [2, 3],
            [1, 3], [1, 4], [3, 4]], dtype=np.int_)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (np.array([0, 2], dtype=np.int_), np.zeros(2))
        mesh.meshdata['force_bc'] = (np.array([3, 4], dtype=np.int_), np.array([[0, -500], [0, -500]]))

        return mesh 

    ## @ingroup MeshGenerators
    @classmethod
    def from_simple_3d_truss(cls):
        # 单位为英寸 in
        node = np.array([
            [0, 0, 36], [72, 0, 0], 
            [0, 0, -36], [0, 72, 0]], dtype=np.float64)
        cell = np.array([
            [0, 1], [0, 2], [0, 3],
            [1, 2], [1, 3], [2, 3]], dtype=np.int_)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (np.array([6, 7, 8, 9], dtype=np.int_), np.zeros(3))
        mesh.meshdata['force_bc'] = (np.array([0, 1], dtype=np.int_), np.array([0, 900, 0]))

        return mesh 


    ## @ingroup MeshGenerators
    @classmethod
    def generate_cantilevered_mesh(cls):
        # Unit m
        node = np.array([
            [0], [5], [7.5]], dtype=np.float64)
        cell = np.array([
            [0, 1], [1, 2]], dtype=np.int_)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (np.array([0, 1], dtype = np.int_), np.zeros(2))
        mesh.meshdata['force_bc'] = (np.array([0, 1, 2], dtype = np.int_), 
                                     np.array([[-62500, -52083], [-93750, 39062], [-31250, 13021]], dtype = np.int_))

        return mesh 


    ## @ingroup MeshGenerators
    @classmethod
    def generate_tri_beam_frame_mesh(cls):
        # Unit: m
        node = np.array([
            [0, 0.96], [1.44, 0.96], 
            [0, 0], [1.44, 0]], dtype=np.float64)
        cell = np.array([
            [0, 1], [2, 0], [3, 1]], dtype=np.int_)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (np.array([2, 3], dtype=np.int_), np.zeros(3))
        mesh.meshdata['force_bc'] = (np.array([0, 1], dtype=np.int_), 
                                     np.array([[3000, -3000, -720], 
                                               [0, -3000, 720]], dtype=np.float64))

        return mesh 


    ## @ingroup MeshGenerators
    @classmethod
    def plane_frame(cls):
        # 单位为 m
        node = np.array([[0, 6], [5, 6], [5, 3], [0, 3], [0, 0], [5, 9],
                         [5, 0], [0, 9], [1, 6], [2, 6], [3, 6], [4, 6],
                         [5, 4], [5, 5], [1, 3], [2, 3], [3, 3], [4, 3],
                         [0, 1], [0, 2], [0, 4], [0, 5], [5, 7], [5, 8],
                         [5, 1], [5, 2], [0, 7], [0, 8], [1, 9], [2, 9],
                         [3, 9], [4, 9]])


EdgeMesh.set_ploter('1d')

