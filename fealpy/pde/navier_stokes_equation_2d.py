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

class FlowPastCylinder:
    '''
    @brief 圆柱绕流
    '''
    def __init__(self, eps=1e-12, rho=1, mu=0.001):
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
    
    def _dist_mesh(self, h0):
        from fealpy.geometry import DistDomain2d
        from fealpy.mesh import DistMesh2d
        import numpy as np
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

    def mesh(self, h, method:str='fealpy', device=None):
        if method == 'fealpy':
            node,cell = self._fealpy_mesh(h)
            node = bm.device_put(node, device)
            cell = bm.device_put(cell, device) 
        elif mesh == 'gmesh':
            node,cell = self._gmesh_mesh(h)
        #elif method == 'distmesh':
        #    return self._dist_mesh(h)
        else:
            raise ValueError(f"Unknown method:{method}")
        return TriangleMesh(node, cell)


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
        return (bm.sqrt((x-0.2)**2 + (y-0.2)**2) - 0.05) < self.eps
    
    @cartesian
    def is_wall_boundary(self,p):
        return (bm.abs(p[..., 1] -0.41) < self.eps) | \
               (bm.abs(p[..., 1] ) < self.eps)
    
    @cartesian
    def is_u_boundary(self,p):
        return bm.abs(p[..., 0] - 2.2) > self.eps

    @cartesian
    def u_inflow_dirichlet(self, p):
        x = p[...,0]
        y = p[...,1]
        value = bm.zeros(p.shape, dtype=p.dtype)
        value[...,0] = 1.5*4*y*(0.41-y)/(0.41**2)
        value[...,1] = 0
        return value

