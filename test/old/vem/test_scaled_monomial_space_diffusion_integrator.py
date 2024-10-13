import numpy as np
import matplotlib.pyplot as plt
import pytest
import ipdb 

from fealpy.decorator import cartesian
from fealpy.mesh import PolygonMesh, TriangleMesh
from fealpy.functionspace import ScaledMonomialSpace2d 
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d
from fealpy.vem import ScaledMonomialSpaceDiffusionIntegrator2d

@pytest.mark.parametrize('p', range(6))
def test_polygon_mesh(p):
   tmesh = TriangleMesh.from_one_triangle()
   tmesh.uniform_refine()
   pmesh = PolygonMesh.from_mesh(tmesh)

   space = ScaledMonomialSpace2d(pmesh, p=p)

   mint = ScaledMonomialSpaceMassIntegrator2d()
   H = mint.assembly_cell_matrix(space)

   dint = ScaledMonomialSpaceDiffusionIntegrator2d()
   #ipdb.set_trace()
   G = dint.assembly_cell_matrix(space, H) 

   @cartesian
   def f(x, index):
       gphi = space.grad_basis(x, index=index, p=p)
       return np.einsum('ijkm, ijpm->ijkp', gphi, gphi)

   A = space.integralalg.cell_integral(f, q=p+3)

   np.testing.assert_array_almost_equal(G, A)

   if False:
       fig, axes = plt.subplots()
       pmesh.add_plot(axes)
       pmesh.find_node(axes, showindex=True)
       pmesh.find_cell(axes, showindex=True)
       plt.show()



if __name__ == "__main__":
    test_polygon_mesh(1)
