import numpy as np
import matplotlib.pyplot as plt
import pytest
import ipdb 

from fealpy.mesh import PolygonMesh, TriangleMesh
from fealpy.functionspace import ScaledMonomialSpace2d 
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d


@pytest.mark.parametrize('p', range(6))
def test_polygon_mesh(p):
   tmesh = TriangleMesh.from_one_triangle()
   tmesh.uniform_refine()
   pmesh = PolygonMesh.from_mesh(tmesh)

   cell, cellLocation = pmesh.entity('cell')

   space = ScaledMonomialSpace2d(pmesh, p=p)

   inte = ScaledMonomialSpaceMassIntegrator2d()

   H = inte.assembly_cell_matrix(space)
   H0 = space.matrix_H()

   np.testing.assert_array_almost_equal(H, H0)

   if False:
       fig, axes = plt.subplots()
       pmesh.add_plot(axes)
       pmesh.find_node(axes, showindex=True)
       pmesh.find_cell(axes, showindex=True)
       plt.show()



if __name__ == "__main__":
    test_polygon_mesh(1)






