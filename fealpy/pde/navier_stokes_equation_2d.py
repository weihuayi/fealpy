#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: navier_stokes_equation_2d.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Mon 14 Oct 2024 04:53:51 PM CST
	@bref 
	@ref 
'''  
import numpy as np
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

        node = bm.array(mesh.points, dtype=bm.float64)
        cell = bm.array(mesh.elements, dtype=bm.int32)
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

