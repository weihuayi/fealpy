
from fealpy.backend import backend_manager as bm


from fealpy.mesh import HexahedronMesh 
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import LinearElasticityLFEMSolver
from fealpy.material.elastic_material import LinearElasticMaterial



mesh = HexahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=10, ny=10, nz=10)
material = LinearElasticMaterial('hexmesh', elastic_modulus=2.06e5, poisson_ratio=0.3)
s0 = LinearElasticityLFEMSolver(material, mesh, p=1)
