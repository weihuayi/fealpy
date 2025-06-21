import numpy as np
from fealpy.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh 
from fealpy.geometry import SphereSurface
import pytest

@pytest.mark.parametrize('p', range(1, 7)) 
def test_init(p):
    # Define sample node and cell arrays

    surface = SphereSurface() # 以原点为球心，1 为半径的球

    node, cell = surface.init_mesh(meshtype='tri', returnnc=True)

    mesh = LagrangeTriangleMesh(node, cell, surface=surface, p=p)

    node = mesh.entity('node')
    cell = mesh.entity('cell')

    NN = mesh.number_of_nodes()
    NC = mesh.number_of_cells()
    GD = mesh.geo_dimension()
    TD = mesh.top_dimension()

    fname = f"test_{p}.vtu"
    mesh.to_vtk(fname=fname)
    
    assert mesh.itype == cell.dtype
    assert mesh.ftype == node.dtype

    assert mesh.meshtype == 'ltri'
    assert (p+1)*(p+2)//2 == cell.shape[1]
    assert NN == node.shape[0]
    assert NC == cell.shape[0]
    assert GD == node.shape[1]
    assert mesh.nodedata == {}
    assert mesh.edgedata == {}
    assert mesh.celldata == {}
    assert mesh.meshdata == {}
    assert TD == 2

if __name__ == "__main__":
   test_init(2) 


