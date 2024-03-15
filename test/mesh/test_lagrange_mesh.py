import numpy as np

from fealpy.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh 
from fealpy.geometry import SphereSurface
from fealpy.mesh.lagrange_mesh import LagrangeMesh
import pytest
 
def test_jacobi_matrix():

    p = 2

    surface = SphereSurface() # 以原点为球心，1为半径的球
     
    node, cell = surface.init_mesh(meshtype='tri', returnnc=True)

    mesh = LagrangeTriangleMesh(node, cell, surface=surface, p=p)

    node = mesh.entity('node')
    cell = mesh.entity('cell')

    fname = f"test.vtu"
    mesh.to_vtk(fname=fname)

    multiIndex = mesh.multi_index_matrix(p=p, etype=2)
    bc = multiIndex / p # 插值点的重心坐标
    print('bc.shape:', bc.shape)

    J, gphi = mesh.jacobi_matrix(bc)
    #print('J:', J)

    
if __name__ == "__main__":
   test_jacobi_matrix() 
