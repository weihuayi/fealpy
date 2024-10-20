from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh2d

from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator

from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from app.soptx.soptx.cases.material_properties import ElasticMaterialProperties, SIMPInterpolation

from app.soptx.soptx.pde.mbb_beam_2d import MBBBeam2dData1
from app.soptx.soptx.pde.short_cantilever_2d import ShortCantilever2dData1

from app.soptx.soptx.solver.fem_solver import FEMSolver
# pde = MBBBeam2dData1()
pde = ShortCantilever2dData1()

extent = pde.domain(xmin=0, xmax=4, ymin=0, ymax=4)
nx, ny = 4, 3
h = [(extent[1] - extent[0]) / nx, (extent[3] - extent[2]) / ny]
origin = [extent[0], extent[2]]
mesh = UniformMesh2d(extent=extent, h=h, origin=origin, 
                    ipoints_ordering='nec', flip_direction='y', device='cpu')
space_C = LagrangeFESpace(mesh=mesh, p=1, ctype='C')
tensor_space_C = TensorFunctionSpace(scalar_space=space_C, shape=(-1, 2))


isBdDof = tensor_space_C.is_boundary_dof(threshold=pde.threshold(), method='interp')
print("---------------------------------------")
