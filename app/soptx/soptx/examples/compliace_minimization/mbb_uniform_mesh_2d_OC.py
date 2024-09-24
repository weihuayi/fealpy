import argparse

from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.mesh.uniform_mesh_2d import UniformMesh2d

from fealpy.experimental.opt import opt_alg_options

from app.soptx.soptx.cases.material_properties import ElasticMaterialProperties, SIMPInterpolation

from app.soptx.soptx.pde.mbb_beam import MBBBeamOneData

from app.soptx.soptx.opt.volume_objective import VolumeConstraint
from app.soptx.soptx.opt.compliance_objective import ComplianceObjective
from app.soptx.soptx.opt.oc_alg import OCAlg


parser = argparse.ArgumentParser(description="MBB 梁上的柔顺度最小化.")

parser.add_argument('--backend', 
                    default='numpy', type=str,
                    help='指定计算的后端类型, 默认为 numpy.')

parser.add_argument('--degree',
                    default=1, type=int,
                    help='Lagrange 有限元空间的次数, 默认为 1.')

parser.add_argument('--nx', 
                    default=60, type=int, 
                    help='x 方向的初始网格单元数, 默认为 60.')

parser.add_argument('--ny',
                    default=20, type=int,
                    help='y 方向的初始网格单元数, 默认为 20.')

parser.add_argument('--filter_type', 
                    default='density', type=str, 
                    help='滤波器类型, 默认为密度滤波器.')

parser.add_argument('--filter_rmin', 
                    default=1.5, type=float, 
                    help='滤波器半径, 默认为 1.5.')

parser.add_argument('--volfrac', 
                    default=0.5, type=float, 
                    help='体积分数, 默认为 0.5.')

args = parser.parse_args()

bm.set_backend(args.backend)

nx, ny = args.nx, args.ny
pde = MBBBeamOneData(nx=nx, ny=ny)

extent = pde.domain()
h = [(extent[1] - extent[0]) / nx, (extent[3] - extent[2]) / ny]
origin = [extent[0], extent[2]]

mesh = UniformMesh2d(extent=extent, h=h, origin=origin, flip_direction=True)

volfrac = args.volfrac
rho = volfrac * bm.ones(nx * ny, dtype=bm.float64)
# rho[0] = 0.8
# rho[-1] = 1.5

material_properties = ElasticMaterialProperties(
            E0=1.0, Emin=1e-9, nu=0.3, penal=3.0, 
            hypo="plane_stress", rho=rho,
            interpolation_model=SIMPInterpolation())

volume_constraint = VolumeConstraint(mesh=mesh,
                                    volfrac=args.volfrac,
                                    filter_type=args.filter_type,
                                    filter_rmin=args.filter_rmin)

compliance_objective = ComplianceObjective(
    mesh=mesh,
    space_degree=args.degree,
    dof_per_node=2,
    dof_ordering='gd-priority', 
    material_properties=material_properties,
    filter_type=args.filter_type,
    filter_rmin=args.filter_rmin,
    pde=pde,
    solver_method='cg', 
    volume_constraint=volume_constraint
)

options = opt_alg_options(
    x0=material_properties.rho,
    objective=compliance_objective,
    MaxIters=200,
    FunValDiff=0.01
)

oc_optimizer = OCAlg(options)
oc_optimizer.run()