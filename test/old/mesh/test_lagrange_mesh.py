import numpy as np

from fealpy.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh 
from fealpy.geometry import SphereSurface
from fealpy.mesh.lagrange_mesh import LagrangeMesh
import pytest
import ipdb

p = 3

surface = SphereSurface() # 以原点为球心，1为半径的球
 
node, cell = surface.init_mesh(meshtype='tri', returnnc=True)

mesh = LagrangeTriangleMesh(node, cell, surface=surface, p=p)

node = mesh.entity('node')
cell = mesh.entity('cell')

fname = f"test.vtu"
mesh.to_vtk(fname=fname)

#ipdb.set_trace()
multiIndex = mesh.multi_index_matrix(p, etype=2)
bc = multiIndex / p # 插值点的重心坐标
print('bc:', bc)

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

def test_jacobi_matrix():

    J = mesh.jacobi_matrix(bc)
    #print('J', J)
    print('J.shape:', J.shape)

def test_unit_normal():
    n = mesh.unit_normal(bc)
    #print('n:', n)
    print('n.shape:', n.shape)

def test_first_fundamental_form():

    G = mesh.first_fundamental_form(bc)
    #print('G:', G)
    print('G.shape:', G.shape)


if __name__ == "__main__":
   test_jacobi_matrix()
   test_unit_normal()
   test_first_fundamental_form()
