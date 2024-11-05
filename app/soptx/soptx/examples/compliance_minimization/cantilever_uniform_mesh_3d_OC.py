import argparse

from fealpy.backend import backend_manager as bm

from fealpy.mesh.uniform_mesh_3d import UniformMesh3d

from fealpy.opt import opt_alg_options

from app.soptx.soptx.material.material_properties import ElasticMaterialProperties, SIMPInterpolation

from app.soptx.soptx.pde.cantilever_3d import Cantilever3dOneData

from app.soptx.soptx.opt.volume_objective import VolumeConstraint
from app.soptx.soptx.opt.compliance_objective import ComplianceObjective
from app.soptx.soptx.opt.oc_alg import OCAlg


parser = argparse.ArgumentParser(description="短悬臂梁上的柔顺度最小化.")

parser.add_argument('--backend', 
                    default='numpy', type=str,
                    help='指定计算的后端类型, 默认为 numpy.')

parser.add_argument('--degree',
                    default=1, type=int,
                    help='Lagrange 有限元空间的次数, 默认为 1.')

parser.add_argument('--nx', 
                    default=2, type=int, 
                    help='x 方向的初始网格单元数, 默认为 60.')

parser.add_argument('--ny',
                    default=1, type=int,
                    help='y 方向的初始网格单元数, 默认为 20.')

parser.add_argument('--nz',
                    default=2, type=int,
                    help='z 方向的初始网格单元数, 默认为 4.')

parser.add_argument('--filter_type', 
                    default='density', type=str, 
                    help='滤波器类型, 默认为密度滤波器.')

parser.add_argument('--filter_rmin', 
                    default=1.5, type=float, 
                    help='滤波器半径, 默认为 1.5.')

parser.add_argument('--volfrac', 
                    default=0.3, type=float, 
                    help='体积分数, 默认为 0.3.')

args = parser.parse_args()

bm.set_backend(args.backend)

nx, ny, nz = args.nx, args.ny, args.nz
pde = Cantilever3dOneData(nx=nx, ny=ny, nz=nz)

extent = [0, nx, 0, ny, 0, nz]
h = [(extent[1] - extent[0]) / nx, 
     (extent[3] - extent[2]) / ny, 
     (extent[5] - extent[4]) / nz]
origin = [extent[0], extent[2], extent[4]]

mesh = UniformMesh3d(extent=extent, h=h, origin=origin, flip_direction='y')

# import matplotlib.pyplot as plt
# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# plt.show()

volfrac = args.volfrac
NC = mesh.number_of_cells()
rho = volfrac * bm.ones(NC, dtype=bm.float64)

material_properties = ElasticMaterialProperties(
            E0=1.0, Emin=1e-9, nu=0.3, penal=3.0, 
            hypo="3D", rho=rho,
            interpolation_model=SIMPInterpolation())

volume_constraint = VolumeConstraint(mesh=mesh,
                                    volfrac=args.volfrac,
                                    filter_type=args.filter_type,
                                    filter_rmin=args.filter_rmin)

compliance_objective = ComplianceObjective(
    mesh=mesh,
    space_degree=args.degree,
    dof_per_node=3,
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