from typing import Sequence
from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TriangleMesh, UniformMesh
from ..backend import TensorLike

class InterfaceFittedMesher2d:
    """Box domain interface fitted mesh generator"""
    def __init__(self, box=None):
        if box is None:
            self.box = [1, 10, -10, 10]  
        else:
            self.box = box

    def geo_dimension(self) -> int:
        return 2
    
    def domain(self) -> Sequence[float]:
        return self.box
    
    def level_function(self, p: TensorLike) -> TensorLike:
        """Check if point is in Ω+ or Ω-."""
        x = p[...,0]
        y = p[...,1]

        return x**2 + y**2 - 2.1**2

    @variantmethod('uniform_tri')
    def init_mesh(self, nx=10, ny=10, level_function=None):
        hx = (self.box[1] - self.box[0])/nx
        hy = (self.box[3] - self.box[2])/ny

        back_mesh = UniformMesh((0, nx, 0, ny), (hx, hy), (self.box[0], self.box[2]))
           
        if level_function is None:
            level_function = self.level_function
        
        mesh = TriangleMesh.interfacemesh_generator(back_mesh, level_function)
        return mesh, back_mesh