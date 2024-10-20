from fealpy.backend import backend_manager as bm

from fealpy.mesh.uniform_mesh_2d import UniformMesh2d

from fealpy.opt.objective import Objective

from app.soptx.soptx.opt.compliance_objective import ComplianceObjective
from app.soptx.soptx.opt.volume_objective import VolumeConstraint

from app.soptx.soptx.cases.material_properties import ElasticMaterialProperties, SIMPInterpolation

from app.soptx.soptx.pde.mbb_beam_2d import MBBBeam2dOneData

bm.set_backend('numpy')

nx, ny = 2
extent = [0, nx, 0, ny]
h = [1.0, 1.0]
origin = [0.0, 0.0]
mesh = UniformMesh2d(extent=extent, h=h, origin=origin, 
                    ipoints_ordering='yx', flip_direction='y', 
                    device='cpu')

pde = MBBBeam2dOneData(nx=nx, ny=ny)

volfrac = 0.5
rho = volfrac * bm.ones(nx * ny, 
                    dtype=bm.float64, device=bm.get_device(mesh))
material_properties = ElasticMaterialProperties(
            E0=1.0, Emin=1e-9, nu=0.3, penal=3.0, 
            hypo="plane_stress", rho=rho,
            interpolation_model=SIMPInterpolation())

volume_constraint = VolumeConstraint(mesh=mesh, volfrac=volfrac,
                                    filter_type=None, filter_rmin=None) 

compliance_objective = ComplianceObjective(
    mesh=mesh,
    space_degree=1,
    dof_per_node=2,
    dof_ordering='gd-priority', 
    material_properties=material_properties,
    pde=pde,
    solver_method='cg', 
    volume_constraint=volume_constraint,
    filter_type=None, filter_rmin=None
)

dce = compliance_objective.jac(rho=rho)
print("--------------")

