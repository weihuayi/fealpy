import numpy as np
import matplotlib.pyplot as plt
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.mesh import PolygonMesh
from fealpy.functionspace import ScaledMonomialSpace2d

#from scaled_monomial_space_data import *

class TestScaledMonomialSpace2d():
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data",1 )
    def test_top(self, backend, data): 
        bm.set_backend(backend)
        tmesh = TriangleMesh.from_box([0,1,0,1],nx=2,ny=2)
        mesh = PolygonMesh.from_triangle_mesh_by_dual(tmesh)
        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        plt.show()
 
if __name__ == '__main__':
    ts = TestScaledMonomialSpace2d()
    ts.test_top('numpy', 1)
    ts.test_top('pytorch', 1)























