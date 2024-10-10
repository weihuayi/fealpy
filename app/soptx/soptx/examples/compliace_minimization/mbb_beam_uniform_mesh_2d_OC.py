import argparse

from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.mesh.uniform_mesh_2d import UniformMesh2d

from fealpy.experimental.opt import opt_alg_options

from app.soptx.soptx.cases.material_properties import ElasticMaterialProperties, SIMPInterpolation

from app.soptx.soptx.pde.mbb_beam_2d import MBBBeam2dOneData

from app.soptx.soptx.opt.volume_objective import VolumeConstraint
from app.soptx.soptx.opt.compliance_objective import ComplianceObjective
from app.soptx.soptx.opt.oc_alg import OCAlg

parameter_groups = {
    'group1': {'nx': 60, 'ny': 20, 'filter_rmin': 2.4},
    'group2': {'nx': 150, 'ny': 50, 'filter_rmin': 6.0},
    'group3': {'nx': 300, 'ny': 100, 'filter_rmin': 16.0},
}

parser = argparse.ArgumentParser(description="MBB 梁上的柔顺度最小化.")

parser.add_argument('--backend', 
                    default='numpy', type=str,
                    help='指定计算的后端类型, 默认为 numpy.')

parser.add_argument('--degree',
                    default=1, type=int,
                    help='Lagrange 有限元空间的次数, 默认为 1.')

parser.add_argument('--solver_method',
                    default='spsolve', type=str,
                    help='求解器类型, 默认为 cg.')

parser.add_argument('--filter_type', 
                    default='sensitivity', type=str, 
                    help='滤波器类型, 默认为密度滤波器.')

parser.add_argument('--volfrac', 
                    default=0.5, type=float, 
                    help='体积分数, 默认为 0.5.')

parser.add_argument('--group', 
                    choices=parameter_groups.keys(), 
                    default='group1',
                    help=(
                        '选择参数组 (例如 group1, group2, group3 等).\n'
                        '每个参数组定义如下:\n'
                        'nx: x 方向的单元数\n'
                        'ny: y 方向的单元数\n'
                        'filter_rmin: 滤波器的半径\n'
                    ))

args = parser.parse_args()

bm.set_backend(args.backend)

args_group = parameter_groups[args.group]

nx, ny = args_group['nx'], args_group['ny']
pde = MBBBeam2dOneData(nx=nx, ny=ny)

extent = [0, nx, 0, ny]
h = [1.0, 1.0]
origin = [0.0, 0.0]

mesh = UniformMesh2d(extent=extent, h=h, origin=origin, flip_direction='y')
# import matplotlib.pyplot as plt
# fig = plt.figure()
# axes = fig.add_subplot(111)
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# plt.show()

volfrac = args.volfrac
rho = volfrac * bm.ones(nx * ny, dtype=bm.float64)

material_properties = ElasticMaterialProperties(
            E0=1.0, Emin=1e-9, nu=0.3, penal=3.0, 
            hypo="plane_stress", rho=rho,
            interpolation_model=SIMPInterpolation())

volume_constraint = VolumeConstraint(mesh=mesh,
                                    volfrac=args.volfrac,
                                    filter_type=args.filter_type,
                                    filter_rmin=args_group['filter_rmin']) 

compliance_objective = ComplianceObjective(
    mesh=mesh,
    space_degree=args.degree,
    dof_per_node=2,
    dof_ordering='gd-priority', 
    material_properties=material_properties,
    filter_type=args.filter_type,
    filter_rmin=args_group['filter_rmin'],
    pde=pde,
    solver_method=args.solver_method, 
    volume_constraint=volume_constraint
)

options = opt_alg_options(
    x0=material_properties.rho,
    objective=compliance_objective,
    MaxIters=10,
    FunValDiff=0.01
)

oc_optimizer = OCAlg(options)
oc_optimizer.run()