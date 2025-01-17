#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: navier_stokes_equation_2d.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Mon 14 Oct 2024 04:53:51 PM CST
	@bref 
	@ref 
'''  
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
class ChannelFlow: 
    def __init__(self, eps=1e-10, rho=1, mu=1, R=None):
        self.eps = eps
        self.rho = rho
        self.mu = mu
        if R is None:
            self.R = rho/mu
    
    def mesh(self, n=16):
        box = [0, 1, 0, 1]
        mesh = TriangleMesh.from_box(box, nx=n, ny=n)
        return mesh
    
    @cartesian
    def velocity(self, p):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros(p.shape)
        value[...,0] = 4*y*(1-y)
        return value
    
    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        val = 8*(1-x) 
        return val
    
    @cartesian
    def is_p_boundary(self, p):
        tag_left = bm.abs(p[..., 0]) < self.eps
        tag_right = bm.abs(p[..., 0] - 1.0) < self.eps
        return tag_left | tag_right

    @cartesian
    def is_u_boundary(self, p):
        tag_up = bm.abs(p[..., 1] - 1.0) < self.eps
        tag_down = bm.abs(p[..., 1] - 0.0) < self.eps
        return tag_up | tag_down 
    
class FlowPastCylinder:
    '''
    @brief 圆柱绕流
    '''
    def __init__(self, eps=1e-10, rho=1, mu=0.001):
        self.eps = eps
        self.rho = rho
        self.mu = mu
    
    def _gmesh_mesh(self, h): 
        import gmsh
        gmsh.initialize()

        gmsh.model.add("gntest2")
        lc = h

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
        return node, cell
    
    def _fealpy_mesh(self,h):
        from meshpy.triangle import MeshInfo, build
        from fealpy.mesh import IntervalMesh
        backend = bm.backend_name
        bm.set_backend('numpy')
        points = bm.array([[0.0, 0.0], [2.2, 0.0], [2.2, 0.41], [0.0, 0.41]],
                dtype=bm.float64)
        facets = bm.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=bm.int32)


        mm = IntervalMesh.from_circle_boundary([0.2, 0.2], 0.1, int(2*0.1*bm.pi/0.01))
        p = mm.entity('node')
        f = mm.entity('cell')

        points = bm.concat((points, p), axis=0)
        facets = bm.concat((facets, f+4), axis=0)
        
        mesh_info = MeshInfo()
        mesh_info.set_points(bm.to_numpy(points))
        mesh_info.set_facets(bm.to_numpy(facets))
        mesh_info.set_holes([[0.2, 0.2]])
        mesh = build(mesh_info, max_volume=h**2)
        node = bm.array(mesh.points, dtype=bm.float64)
        cell = bm.array(mesh.elements, dtype=bm.int32)
        
        bm.set_backend(backend)
        node = bm.from_numpy(node)
        cell = bm.from_numpy(cell)
        return node,cell

    def mesh(self, h, method:str='fealpy', device='cpu'):
        if method == 'fealpy':
            node,cell = self._fealpy_mesh(h)
            node = bm.device_put(node, device)
            cell = bm.device_put(cell, device) 
        elif mesh == 'gmesh':
            node,cell = self._gmesh_mesh(h)
        else:
            raise ValueError(f"Unknown method:{method}")
        return TriangleMesh(node, cell)


    @cartesian
    def is_outflow_boundary(self,p):
        x = p[...,0]
        y = p[...,1]
        cond1 = bm.abs(x - 2.2) < self.eps
        cond2 = bm.abs(y-0)>self.eps
        cond3 = bm.abs(y-0.41)>self.eps
        return (cond1) & (cond2 & cond3) 
    
    @cartesian
    def is_inflow_boundary(self,p):
        return bm.abs(p[..., 0]) < self.eps
    
    @cartesian
    def is_circle_boundary(self,p):
        x = p[...,0]
        y = p[...,1]
        return (bm.sqrt((x-0.2)**2 + (y-0.2)**2) - 0.05) < self.eps
    
    @cartesian
    def is_wall_boundary(self,p):
        return (bm.abs(p[..., 1] -0.41) < self.eps) | \
               (bm.abs(p[..., 1] ) < self.eps)
    
    @cartesian
    def is_u_boundary(self,p):
        return ~self.is_outflow_boundary(p)
    
    @cartesian
    def is_p_boundary(self,p):
        return self.is_outflow_boundary(p) 

    @cartesian
    def u_inflow_dirichlet(self, p):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros_like(p)
        value[...,0] = 1.5*4*y*(0.41-y)/(0.41**2)
        value[...,1] = 0
        return value
    
    @cartesian
    def p_dirichlet(self, p):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros_like(x)
        return value

    @cartesian
    def u_dirichlet(self, p):
        x = p[...,0]
        y = p[...,1]
        index = self.is_inflow_boundary(p)
        result = bm.zeros_like(p)
        result[index] = self.u_inflow_dirichlet(p[index])
        return result
