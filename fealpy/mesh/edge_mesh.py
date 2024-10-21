from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .mesh_base import SimplexMesh
from .plot import Plotable


class EdgeMesh(SimplexMesh, Plotable):
    """
    """
    def __init__(self, node, cell):
        super().__init__(TD=1, itype=cell.dtype, ftype=node.dtype)
        self.node = node
        self.cell = cell

        self.nodedata = {}
        self.facedata = self.nodedata
        self.celldata = {}
        self.edgedata = self.celldata 
        self.meshdata = {}

        self.cell_length = self.edge_length
        self.cell_tangent = self.edge_tangent
        

    def ref_cell_measure(self):
        return 1.0
    
    def ref_face_measure(self):
        return 0.0
    
    def quadrature_formula(self, q: int, etype: Union[str, int]='cell'):
        """
        @brief 返回第 k 个高斯积分公式。
        """
        from ..quadrature import GaussLegendreQuadrature
        return GaussLegendreQuadrature(q, dtype=self.ftype, device=self.device)

    def edge_tangent(self,index = None):
        edge = self.entity('edge', index=index)
        node = self.entity('node')
        return bm.edge_tangent(edge, node)

    def edge_length(self, index=None):
        """
        @brief 计算边的长度
        """
        edge = self.entity('edge', index=index)
        node = self.entity('node')
        return bm.edge_length(edge, node)
    
    def entity_measure(self, etype: Union[int, str]='cell', index=None, node=None):
        """
        """
        if etype in {1, 'cell', 'edge'}:
            return self.cell_length(index=index)
        elif etype in {0, 'face', 'node'}:
            return bm.tensor([0.0], dtype=self.ftype)
        else:
            raise ValueError(f"entity type: {etype} is wrong!")
        
    def grad_lambda(self, index=None):
        """
        @brief 计算所有单元上重心坐标函数的导数
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        v = node[cell[:, 1]] - node[cell[:, 0]]
        NC = len(cell) 
        GD = self.geo_dimension()
        h2 = bm.sum(v**2, axis=-1)
        v /=h2.reshape(-1, 1)
        v = v[:,None,:]
        Dlambda = bm.concatenate([-v,v],axis=1)
        return Dlambda
    
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        return p + 1
    
    def number_of_global_ipoints(self, p: int) -> int:
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        return NN + (p-1)*NC
    
    def interpolation_points(self, p: int, index=None):
        GD = self.geo_dimension()
        node = self.entity('node')

        if p == 1:
            return node
        else:
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            gdof = NN + NC*(p-1)
            cell = self.entity('cell')
            a = bm.arange(p-1,0,-1,dtype=bm.float64)/p
            a = a.reshape(p-1,-1)
            b = bm.arange(1,p,1,dtype=bm.float64)/p
            b = b.reshape(p-1,-1)
            w = bm.concatenate([a,b],axis=1)
            GD = self.geo_dimension()
            cip = bm.einsum('ij, kj...->ki...', w,node[cell]).reshape(-1, GD)
            ipoint = bm.concatenate([node,cip],axis=0)
            return ipoint
    
    def face_unit_normal(self, index=None, node=None):
        """
        @brief
        """
        raise NotImplementedError

    def cell_normal(self, index=None, node=None):
        """
        @brief 单元的法线方向
        """
        assert self.geo_dimension() == 2
        v = self.cell_tangent(index=index)
        w = bm.tensor([(0, -1),(1, 0)],dtype=self.ftype)
        return v@w
    
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
        node = bm.tensor([
            [-950, 0, 5080], [950, 0, 5080], [-950, 950, 2540], 
            [950, 950, 2540], [950, -950, 2540], [-950, -950, 2540],
            [-2540, 2540, 0], [2540, 2540, 0], [2540, -2540, 0], 
            [-2540, -2540, 0]], dtype=bm.float64)
        cell = bm.tensor([
            [0, 1], [3, 0], [1, 2], [1, 5], [0, 4], 
            [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
            [4, 3], [2, 3], [4, 5], [2, 9], [6, 5], 
            [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
            [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=bm.int_)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (bm.tensor([6, 7, 8, 9], dtype=bm.int_), bm.zeros(3))
        mesh.meshdata['force_bc'] = (bm.tensor([0, 1], dtype=bm.int_), bm.tensor([0, 900, 0]))

        return mesh 
    
    ## @ingroup MeshGenerators
    @classmethod
    def from_four_bar_mesh(cls):
        # 单位为 mm
        node = bm.tensor([
            [0, 0], [400, 0], 
            [400, 300], [0, 300]], dtype=bm.float64)
        cell = bm.tensor([
            [0, 1], [2, 1], 
            [0, 2], [3, 2]], dtype=bm.int_)
        mesh = cls(node, cell)

        # 按分量处理自由度索引
        mesh.meshdata['disp_bc'] = (bm.tensor([0, 1, 3, 6, 7], dtype=bm.int_), bm.zeros(1))
        mesh.meshdata['force_bc'] = (bm.tensor([1, 2], dtype=bm.int_), 
                                     bm.tensor([[2e4, 0], [0, -2.5e4]], dtype=bm.float64))

        return mesh 
    
    ## @ingroup MeshGenerators
    @classmethod
    def generate_balcony_truss_mesh(cls):
        # 单位为英寸 in
        node = bm.tensor([
            [0, 0], [36, 0], 
            [0, 36], [36, 36], [72, 36]], dtype=bm.float64)
        cell = bm.tensor([
            [0, 1], [1, 2], [2, 3],
            [1, 3], [1, 4], [3, 4]], dtype=bm.int_)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (bm.tensor([0, 2], dtype=bm.int_), bm.zeros(2))
        mesh.meshdata['force_bc'] = (bm.tensor([3, 4], dtype=bm.int_), bm.tensor([[0, -500], [0, -500]]))

        return mesh 
    
    ## @ingroup MeshGenerators
    @classmethod
    def from_simple_3d_truss(cls):
        # 单位为英寸 in
        node = bm.tensor([
            [0, 0, 36], [72, 0, 0], 
            [0, 0, -36], [0, 72, 0]], dtype=bm.float64)
        cell = bm.tensor([
            [0, 1], [0, 2], [0, 3],
            [1, 2], [1, 3], [2, 3]], dtype=bm.int_)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (bm.tensor([6, 7, 8, 9], dtype=bm.int_), bm.zeros(3))
        mesh.meshdata['force_bc'] = (bm.tensor([0, 1], dtype=bm.int_), bm.tensor([0, 900, 0]))

        return mesh 
    
    ## @ingroup MeshGenerators
    @classmethod
    def generate_cantilevered_mesh(cls):
        # Unit m
        node = bm.tensor([
            [0], [5], [7.5]], dtype=bm.float64)
        cell = bm.tensor([
            [0, 1], [1, 2]], dtype=bm.int_)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (bm.tensor([0, 1], dtype = bm.int_), bm.zeros(2))
        mesh.meshdata['force_bc'] = (bm.tensor([0, 1, 2], dtype = bm.int_), 
                                     bm.tensor([[-62500, -52083], [-93750, 39062], [-31250, 13021]], dtype = bm.int_))

        return mesh 
    
    ## @ingroup MeshGenerators
    @classmethod
    def generate_tri_beam_frame_mesh(cls):
        # Unit: m
        node = bm.tensor([
            [0, 0.96], [1.44, 0.96], 
            [0, 0], [1.44, 0]], dtype=bm.float64)
        cell = bm.tensor([
            [0, 1], [2, 0], [3, 1]], dtype=bm.int_)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (bm.tensor([2, 3], dtype=bm.int_), bm.zeros(3))
        mesh.meshdata['force_bc'] = (bm.tensor([0, 1], dtype=bm.int_), 
                                     bm.tensor([[3000, -3000, -720], 
                                               [0, -3000, 720]], dtype=bm.float64))

        return mesh 
    
    ## @ingroup MeshGenerators
    @classmethod
    def plane_frame(cls):
        # 单位为 m
        node = bm.tensor([[0, 6], [5, 6], [5, 3], [0, 3], [0, 0], [5, 9],
                         [5, 0], [0, 9], [1, 6], [2, 6], [3, 6], [4, 6],
                         [5, 4], [5, 5], [1, 3], [2, 3], [3, 3], [4, 3],
                         [0, 1], [0, 2], [0, 4], [0, 5], [5, 7], [5, 8],
                         [5, 1], [5, 2], [0, 7], [0, 8], [1, 9], [2, 9],
                         [3, 9], [4, 9]])


EdgeMesh.set_ploter('1d')
