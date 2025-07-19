from typing import Sequence
from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TriangleMesh, UniformMesh

class InterfaceFittedMesher2d:
    """Box domain interface fitted mesh generator"""
    def __init__(self, box=None):
        if box is None:
            self.box = [0, 1, 0, 1]
        else:
            self.box = box

    def geo_dimension(self) -> int:
        return 2
    
    def domain(self) -> Sequence[float]:
        return self.box

    @variantmethod('uniform_tri')
    def init_mesh(self, nx=10, ny=10, level_function=None):
        if level_function is None:
            mesh = TriangleMesh.from_box(box=self.box, nx=nx, ny=ny)
            return mesh
        else:
            back_mesh = UniformMesh((0, nx, 0, ny), h=((self.box[1] - self.box[0])/nx, (self.box[3] - self.box[2])/ny), origin=(self.box[0], self.box[2]))
            mesh = TriangleMesh.interfacemesh_generator(back_mesh, level_function)
            return mesh