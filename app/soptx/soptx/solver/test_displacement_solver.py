from fealpy.backend import backend_manager as bm

from fealpy.mesh import UniformMesh2d

from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator

from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from app.soptx.soptx.cases.material_properties import ElasticMaterialProperties, SIMPInterpolation
from app.soptx.soptx.pde.mbb_beam_2d import MBBBeam2dData1
from app.soptx.soptx.solver.fem_solver import FEMSolver

bm.set_backend('numpy')

pde = MBBBeam2dData1()
extent = pde.domain()
nx, ny = extent[1], extent[3]
h = [(extent[1] - extent[0]) / nx, (extent[3] - extent[2]) / ny]
origin = [extent[0], extent[2]]
mesh = UniformMesh2d(extent=extent, h=h, origin=origin, 
                    ipoints_ordering='nec', flip_direction='y', device='cpu')
space_C = LagrangeFESpace(mesh=mesh, p=1, ctype='C')

tensor_space_C = TensorFunctionSpace(scalar_space=space_C, shape=(-1, 2))

volfrac = 0.5
rho = volfrac * bm.ones(nx * ny, dtype=bm.float64, device=bm.get_device(mesh))


material_properties = ElasticMaterialProperties(
            E0=1.0, Emin=1e-9, nu=0.3, penal=3.0, 
            hypo="plane_stress", rho=rho, 
            interpolation_model=SIMPInterpolation(), 
            device=bm.get_device(mesh))



solver = FEMSolver(
            material_properties=material_properties, 
            tensor_space=tensor_space_C, 
            pde=pde)

uh = solver.solve(solver_method='cg')



integrator = LinearElasticIntegrator(material=material_properties, q=tensor_space_C.p+3)


D = material_properties.elastic_matrix()

KE = integrator.assembly(space=tensor_space_C)