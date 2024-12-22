from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import InteriorPenaltyFESpace2d

from fealpy.fem import ScalarBiharmonicIntegrator, BilinearForm, LinearForm, ScalarInteriorPenaltyIntegrator



bm.set_backend('pytorch')
mesh = TriangleMesh.from_box([0,1,0,1],1,1)
space = LagrangeFESpace(mesh, p=2)
ipspace = InteriorPenaltyFESpace2d(mesh, p=2)

bi = ScalarBiharmonicIntegrator()
assembly_cell_matrix = bi.assembly(space)
#print(assembly_cell_matrix)

ip = ScalarInteriorPenaltyIntegrator()
assembly_edge_matrix = ip.assembly(ipspace)
print(assembly_edge_matrix)

