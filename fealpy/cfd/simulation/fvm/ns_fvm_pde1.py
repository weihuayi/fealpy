from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh

class NSFVMPde1:
    def __init__(self):
        
        pass
    def mesh(self, nx, ny):
        self.nx = nx
        self.ny = ny
        mesh = QuadrangleMesh.from_box(box=[0,1,0,1], nx=nx, ny=ny)
        self.mesh = mesh
        mesh.error
        
        return mesh
    
    def mesh0(self):

        self.h = 1/self.nx
        h = self.h
        nx0 = self.nx +1
        ny0 = self.ny
        self.nx0 = nx0
        self.ny0 = ny0
        mesh0 = QuadrangleMesh.from_box(box=[-1/2*h, 1+1/2*h, 0, 1],nx=nx0,ny=ny0)
        self.mesh0 = mesh0
        self.c2c0 = mesh0.cell_to_cell()

        return mesh0
    
    def mesh1(self):

        h = self.h
        nx1 = self.nx
        ny1 = self.ny + 1
        self.nx1 = nx1
        self.ny1 = ny1
        mesh1 = QuadrangleMesh.from_box(box=[0, 1, -1/2*h, 1+1/2*h],nx=nx1,ny=ny1)
        self.mesh1 = mesh1
        self.c2c1 = mesh1.cell_to_cell()

        return mesh1

    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape)
        val[..., 1] = -3*x*(2-x)
        return val
    def velocity_u(self, p):
        return self.velocity(p)[..., 0]

    def velocity_v(self, p):
        return self.velocity(p)[..., 1]

    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 3*(2*y-1)
        return val

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape)
        return val
    
    def source_u(self, p):
        return self.source(p)[..., 0]   
    
    def source_v(self, p):
        return self.source(p)[..., 1]

    def u_dirichlet(self, p):
        return self.velocity_u(p)

    def v_dirichlet(self, p):
        return self.velocity_v(p)

    def p_dirichlet(self, p):
        return self.pressure(p)

    def is_boundary(self, p):
        eps = 1e-12
        x = p[..., 0]
        y = p[..., 1]
        return (bm.abs(y-1)<eps)|(bm.abs(x-1)<eps)|(bm.abs(x)<eps)|(bm.abs(y)<eps)

    def is_cell_boundary(self, p):
        h = self.h
        eps = 1e-12
        x = p[..., 0]
        y = p[..., 1]
        return (bm.abs(y - h/2)<eps)|(bm.abs(x - h/2)<eps)|(bm.abs(x - (1 - h/2))<eps)|(bm.abs(y - (1 - h/2))<eps)

    def is_edge_boundary(self, p):
        h = self.h
        eps = 1e-12
        x = p[..., 0]
        y = p[..., 1]
        return (bm.abs(y - h/2)<eps)|(bm.abs(x - h/2)<eps)|(bm.abs(x - (1 - h/2))<eps)|(bm.abs(y - (1 - h/2))<eps)
    
