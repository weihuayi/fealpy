from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian

from typing import Tuple, Callable

class Cantilever2dData1:
    '''
    模型来源论文: Efficient topology optimization in MATLAB using 88 lines of code
    '''
    def __init__(self, 
                xmin: float, xmax: float, 
                ymin: float, ymax: float):
        """
        位移边界条件：梁的左边界固定
        载荷：梁的右边界的下点施加垂直向下的力 F = -1
        flip_direction = True
        0 ------- 3 ------- 6 
        |    0    |    2    |
        1 ------- 4 ------- 7 
        |    1    |    3    |
        2 ------- 5 ------- 8 
        """
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.eps = 1e-12

    def domain(self) -> list:
        
        box = [self.xmin, self.xmax, self.ymin, self.ymax]

        return box
    
    @cartesian
    def force(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        coord = (
            (bm.abs(x - domain[1]) < self.eps) & 
            (bm.abs(y - domain[2]) < self.eps)
        )
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[coord, 1] = -1

        return val
    
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps
        
        return coord    
    
    def threshold(self) -> Tuple[Callable, Callable]:

        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)

class Cantilever2dData2:
    '''
    新模型，适应区域大小和载荷改变：
    载荷施加在右边界的中点位置，大小为 T = 2000
    区域尺寸：L 和 H
    '''
    def __init__(self, 
                 xmin: float = 0, xmax: float = 3.0, 
                 ymin: float = 0, ymax: float = 1.0, 
                 T: float = 2000):
        """
        位移边界条件：梁的左边界固定
        载荷：梁的右边界的中点施加垂直向下的力 T = 2000
        0 ------- 3 ------- 6 
        |    0    |    2    |
        1 ------- 4 ------- 7 
        |    1    |    3    |
        2 ------- 5 ------- 8 
        """
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.T = T  # 载荷大小
        self.eps = 1e-12

    def domain(self) -> list:
        box = [self.xmin, self.xmax, self.ymin, self.ymax]
        return box
    
    @cartesian
    def force(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]
        y = points[..., 1]

        # 载荷施加在右边界的中点处
        coord = (
            (bm.abs(x - domain[1]) < self.eps) & 
            (bm.abs(y - (domain[2] + domain[3]) / 2) < self.eps)  # 位于右边界中点
        )
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[coord, 1] = -self.T  # 施加单位力 T
        
        return val
    
    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:
        # 这里仍然是固定左边界的位移
        return bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
    
    @cartesian
    def is_dirichlet_boundary_dof_x(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps  # 左边界的 x 坐标
        
        return coord
    
    @cartesian
    def is_dirichlet_boundary_dof_y(self, points: TensorLike) -> TensorLike:
        domain = self.domain()

        x = points[..., 0]

        coord = bm.abs(x - domain[0]) < self.eps  # 左边界的 x 坐标
        
        return coord
    
    def threshold(self) -> Tuple[Callable, Callable]:
        return (self.is_dirichlet_boundary_dof_x, 
                self.is_dirichlet_boundary_dof_y)