import numpy as np
import gmsh
import meshio
from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian
#from fealpy.geometry import DistDomain2d
#from fealpy.mesh import DistMesh2d
from fealpy.old.geometry import dcircle,drectangle,ddiff,dmin
from fealpy.old.geometry import huniform
from fealpy.backend import backend_manager as bm
class SinCosData:
    """
    [0, 1]^2
    u(x, y) = (sin(pi*x)*cos(pi*y), -cos(pi*x)*sin(piy))
    p = 1/(y**2 + 1) - pi/4
    """
    def __init__(self):
        self.box = [0, 1, 0, 1]

    def domain(self):
        return self.box

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        cos = np.cos
        sin = np.sin
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = sin(pi*x)*cos(pi*y) 
        val[..., 1] = -cos(pi*x)*sin(pi*y) 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 1/(y**2 + 1) - pi/4 
        return val
    
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        sin = np.sin
        cos = np.cos
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 2*pi**2*sin(pi*x)*cos(pi*y) + pi*sin(pi*x)*cos(pi*x)
        val[..., 1] = -2*y/(y**2 + 1)**2 - 2*pi**2*sin(pi*y)*cos(pi*x) + pi*sin(pi*y)*cos(pi*x) 
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class Poisuille:
    """
    [0, 1]^2
    u(x, y) = (4y(1-y), 0)
    p = 8(1-x)
    """
    def __init__(self):
        self.box = [0, 1, 0, 1]
        self.eps = 1e-10
        self.rho = 1
        self.mu = 1

    def domain(self):
        return self.box

    @cartesian
    def velocity(self, p):
        x = p[...,0]
        y = p[...,1]
        value = np.zeros(p.shape)
        value[...,0] = 4*y*(1-y)
        return value

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = 8*(1-x) 
        return val
    
    @cartesian
    def source(self, p):
        val = np.zeros(p.shape)
        return val

    @cartesian
    def is_p_boundary(self, p):
        return (np.abs(p[..., 0]) < self.eps) | (np.abs(p[..., 0] - 1.0) < self.eps)
      
    @cartesian
    def is_wall_boundary(self, p):
        return (np.abs(p[..., 1]-0.0) < self.eps) | (np.abs(p[..., 1] - 1.0) < self.eps)

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

class FlowPastCylinder:
    '''
    @brief 圆柱绕流
    '''
    def __init__(self, eps=1e-12, rho=1, mu=0.001):
        self.eps = eps
        self.rho = rho
        self.mu = mu
    
    def mesh2(self, h0):
        fd1 = lambda p: dcircle(p,[0.2,0.2],0.05)
        fd2 = lambda p: drectangle(p,[0.0,2.2,0.0,0.41])
        fd = lambda p: ddiff(fd2(p),fd1(p))

        def fh(p):
            h = 0.003 + 0.05*fd1(p)
            h[h>0.01] = 0.01
            return h

        bbox = [0,3,0,1]
        pfix = np.array([(0.0,0.0),(2.2,0.0),(2.2,0.41),(0.0,0.41)],dtype=np.float64)
        domain = DistDomain2d(fd,fh,bbox,pfix)
        distmesh2d = DistMesh2d(domain,h0)
        distmesh2d.run()

        mesh = distmesh2d.mesh
        return mesh



    def mesh1(self,h):
        from meshpy.triangle import MeshInfo, build
        from fealpy.mesh import IntervalMesh
        points = np.array([[0.0, 0.0], [2.2, 0.0], [2.2, 0.41], [0.0, 0.41]],
                dtype=np.float64)
        facets = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int_)


        mm = IntervalMesh.from_circle_boundary([0.2, 0.2], 0.1, int(2*0.1*np.pi/0.01))
        p = mm.entity('node')
        f = mm.entity('cell')

        points = np.append(points, p, axis=0)
        facets = np.append(facets, f+4, axis=0)

        fm = np.array([0, 1, 2, 3])


        mesh_info = MeshInfo()
        mesh_info.set_points(points)
        mesh_info.set_facets(facets)

        mesh_info.set_holes([[0.2, 0.2]])

        mesh = build(mesh_info, max_volume=h**2)

        node = np.array(mesh.points, dtype=np.float64)
        cell = np.array(mesh.elements, dtype=np.int_)
        #mesh = TriangleMesh(node,cell)  
        return node,cell

    def mesh(self): 
        gmsh.initialize()

        gmsh.model.add("gntest2")
        lc = 0.01

        gmsh.model.geo.addPoint(0.0,0.0,0.0,lc,1)
        gmsh.model.geo.addPoint(2.2,0.0,0.0,lc,2)
        gmsh.model.geo.addPoint(2.2,0.41,0.0,lc,3)
        gmsh.model.geo.addPoint(0.0,0.41,0.0,lc,4)

        gmsh.model.geo.addLine(1,2,1)
        gmsh.model.geo.addLine(2,3,2)
        gmsh.model.geo.addLine(3,4,3)
        gmsh.model.geo.addLine(4,1,4)

        gmsh.model.geo.addPoint(0.15,0.2,0.0,lc,5)
        gmsh.model.geo.addPoint(0.25,0.2,0.0,lc,6)
        gmsh.model.geo.addPoint(0.2,0.2,0.0,lc,7)

        gmsh.model.geo.add_circle_arc(6,7,5,5)
        gmsh.model.geo.add_circle_arc(5,7,6,6)

        gmsh.model.geo.addCurveLoop([1,2,3,4],1)
        gmsh.model.geo.add_curve_loop([5,6],2)

        gmsh.model.geo.addPlaneSurface([1,2],1)
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.field.add("Distance",1)
        gmsh.model.mesh.field.setNumbers(1,"CurvesList",[5,6])
        gmsh.model.mesh.field.setNumber(1,"Sampling",100)

        gmsh.model.mesh.field.add("Threshold",2)
        gmsh.model.mesh.field.setNumber(2,"InField",1)
        gmsh.model.mesh.field.setNumber(2,"SizeMin",0.01)
        gmsh.model.mesh.field.setNumber(2,"SizeMax",0.01)
        gmsh.model.mesh.field.setNumber(2,"DistMin",0.01)
        gmsh.model.mesh.field.setNumber(2,"DistMax",0.01)

        #gmsh.model.mesh.field.setAsBackgroundMesh(2)
        gmsh.option.setNumber('Mesh.Algorithm',6) 
        gmsh.model.mesh.generate(2)
        gmsh.write("gn.msh")
        mesh = meshio.read('gn.msh',file_format = 'gmsh')
        node = mesh.points[:,:2]
        cell = mesh.cells_dict['triangle']
        mesh = TriangleMesh(node,cell)
        return mesh
    
    @cartesian
    def is_outflow_boundary(self,p):
        return bm.abs(p[..., 0] - 2.2) < self.eps
    
    @cartesian
    def is_inflow_boundary(self,p):
        return bm.abs(p[..., 0]) < self.eps
    
    @cartesian
    def is_circle_boundary(self,p):
        x = p[...,0]
        y = p[...,1]
        return (bm.sqrt(x**2 + y**2) - 0.05) < self.eps
      
    @cartesian
    def is_wall_boundary(self,p):
        return (bm.abs(p[..., 1] -0.41) < self.eps) | \
               (bm.abs(p[..., 1] ) < self.eps)

    @cartesian
    def u_inflow_dirichlet(self, p):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros(p.shape, dtype=p.dtype)
        value[...,0] = 1.5*4*y*(0.41-y)/(0.41**2)
        value[...,1] = 0
        return value
    
class ChannelFlowWithLevelSet:
    def __init__(self, domain ,eps=1e-12, rho=1, mu=0.001):
        self.eps = eps
        self.rho = rho
        self.mu = mu
        self.domain = domain
    
    def mesh(self, nx ,ny):
        domain = self.domain
        mesh = MF.boxmesh2d([domain[0],domain[1],domain[2],domain[3]], nx, ny)
        return mesh
    
    @cartesian
    def is_outflow_boundary(self,p):
        domain = self.domain
        return np.abs(p[..., 0] - domain[1]) < self.eps
    
    @cartesian
    def is_inflow_boundary(self,p):
        domain = self.domain
        return np.abs(p[..., 0] - domain[0]) < self.eps
    
      
    @cartesian
    def is_wall_boundary(self,p):
        domain = self.domain
        return (np.abs(p[..., 1] - domain[2]) < self.eps) | \
               (np.abs(p[..., 1] - domain[3]) < self.eps)

    @cartesian
    def u_inflow_dirichlet(self, p):
        domain = self.domain
        x = p[...,0]
        y = p[...,1]
        value = np.zeros(p.shape,dtype=np.float)
        value[...,0] = 1.5*4*y*(domain[3]-y)/(domain[3]**2)
        value[...,1] = 0
        return value
    
