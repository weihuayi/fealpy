from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .mesh_base import SimplexMesh


class EdgeMesh(SimplexMesh):
    """
    """
    def __init__(self, node, cell):
        super().__init__(TD=1)
        kwargs = {'dtype': cell.dtype}
        self.node = node
        self.cell = cell

        self.nodedata = {}
        self.facedata = self.nodedata
        self.celldata = {}
        self.edgedata = self.celldata 
        self.meshdata = {}

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
            [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=np.int_)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (bm.tensor([6, 7, 8, 9], dtype=np.int_), bm.zeros(3))
        mesh.meshdata['force_bc'] = (bm.tensor([0, 1], dtype=bm.int_), bm.tensor([0, 900, 0]))

        return mesh 
