from fealpy.typing import Tuple, TensorLike, Index, _S
from fealpy.backend import backend_manager as bm
from fealpy.mesh import EdgeMesh


class ChannelBeamMesh():
    """ Channel beam mesh for finite element analysis.
    This class generates a 1D mesh along the length of a channel beam.
    """
    
    def __init__(self, 
                 length: float, 
                 height: float, 
                 width: float, 
                 num_elements: int, 
                 t1: float, 
                 t2: float):
        """Initialize the ChannelBeamMesh.

        Parameters:
            length (float): Length of the beam.
            height (float): Height of the beam.
            width (float): Width of the beam.
            num_elements (int): Number of elements along the beam length.
            t1 (float): Thickness of the top flange.
            t2 (float): Thickness of the bottom flange.
        """
        self.length = length
        self.height = height
        self.width = width
        self.num_elements = num_elements
        self.t1 = t1
        self.t2 = t2
        self.mesh = self.create_mesh()

    def create_mesh(self):
        """Creates a 1D mesh along the beam length for finite element analysis."""
        
        # 创建 1D 网格
        nodes = bm.linspace(0, self.length, self.num_elements + 1)
        
        
        # 定义槽形横截面，四个点
        # 左下角、右下角、右上角、左上角
        points = bm.array([
            [0.0, 0.0, 0.0],  # 左下角
            [self.width, 0.0, 0.0],  # 右下角
            [self.width - self.t2, self.height, 0.0],  # 右上角
            [self.t1, self.height, 0.0]  # 左上角
        ])
        
        node = bm.zeros((self.num_elements + 1, 3), dtype=bm.float64)
        node[:, 0] = nodes  # x-coordinates along beam length

        cell = bm.zeros((self.num_elements, 2), dtype=bm.int32)
        cell[:, 0] = bm.arange(self.num_elements)
        cell[:, 1] = bm.arange(1, self.num_elements + 1)
        
        return EdgeMesh(node, cell)
    
