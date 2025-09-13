from typing import Sequence
from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import IntervalMesh, UniformMesh

class IntervalMesher:
    """Interval domain mesh generator"""
    def __init__(self, interval=[0, 1]):
        self.interval = interval

    def geo_dimension(self) -> int:
        return 1
    
    @variantmethod('uniform_interval')
    def init_mesh(self, nx=10): 
        mesh = IntervalMesh.from_interval_domain(interval=self.interval, nx=nx)
        return mesh 
      
    @init_mesh.register('uniform')
    def init_mesh(self, nx=30):
        domain = self.interval 
        extent = (0, nx)
        mesh = UniformMesh(domain, extent)
        return mesh