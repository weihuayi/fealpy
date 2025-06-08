from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh
class Channel: 
    def __init__(self, eps=1e-10, rho=1, mu=1, R=None):
        self.eps = eps
        self.rho = rho
        self.mu = mu
        self.mesh = self.set_mesh()
        if R is None:
            self.R = rho/mu
    
    def domain(self):
        return [0, 1, 0, 1]

    def set_mesh(self, n=16):
        box = [0, 1, 0, 1]
        mesh = TriangleMesh.from_box(box, nx=n, ny=n)
        self.mesh = mesh
        return mesh
    
    @cartesian
    def velocity(self, p):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros(p.shape, dtype=bm.float64)
        value[...,0] = 4*y*(1-y)
        return value
    
    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        val = 8*(1-x) 
        return val
    
    @cartesian
    def is_p_boundary(self, p):
        tag_left = bm.abs(p[..., 0] - 0.0) < self.eps
        tag_right = bm.abs(p[..., 0] - 1.0) < self.eps
        return tag_left | tag_right

    @cartesian
    def is_u_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 1.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        return tag_up | tag_down



