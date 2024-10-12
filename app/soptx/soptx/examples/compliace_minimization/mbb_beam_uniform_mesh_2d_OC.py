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
    'group4': {'nx': 300, 'ny': 100, 'filter_rmin': 9},
}

parser = argparse.ArgumentParser(description="Compliance minimization on MBB beam.")

parser.add_argument('--backend', 
                    default='pytorch', type=str,
                    help='Specify the backend type for computation, default is numpy.')

parser.add_argument('--degree',
                    default=1, type=int,
                    help='Degree of the Lagrange finite element space, default is 1.')

parser.add_argument('--solver_method',
                    choices=['cg', 'spsolve'],
                    default='cg', type=str,
                    help='Solver type, can choose iterative solver "cg" \
                    or direct solver "spsolve", default is cg.')

parser.add_argument('--filter_type', 
                    choices=['sensitivity', 'density', 'heaviside', 'None'], 
                    default='None', type=str, 
                    help='Filter type, can choose sensitivity filter, density filter, \
                    Heaviside projection filter, or no filter. \
                    Default is density filter.')

parser.add_argument('--volfrac', 
                    default=0.5, type=float, 
                    help='Volume fraction, default is 0.5.')

parser.add_argument('--group', 
                    choices=parameter_groups.keys(), 
                    default='group1',
                    help=(
                        'Select parameter group.\n'
                        'Each parameter group is defined as follows:\n'
                        'nx: Number of elements in the x direction\n'
                        'ny: Number of elements in the y direction\n'
                        'filter_rmin: Radius of the filter\n'
                    ))

args = parser.parse_args()
bm.set_backend(args.backend)
args_group = parameter_groups[args.group]

nx, ny = args_group['nx'], args_group['ny']
pde = MBBBeam2dOneData(nx=nx, ny=ny)

extent = [0, nx, 0, ny]
h = [1.0, 1.0]
origin = [0.0, 0.0]
mesh = UniformMesh2d(extent=extent, h=h, origin=origin, 
                    ipoints_ordering='yx', flip_direction='y', 
                    device='cpu')

volfrac = args.volfrac
rho = volfrac * bm.ones(nx * ny, dtype=bm.float64)

material_properties = ElasticMaterialProperties(
            E0=1.0, Emin=1e-9, nu=0.3, penal=3.0, 
            hypo="plane_stress", rho=rho,
            interpolation_model=SIMPInterpolation())

filter_type = args.filter_type
filter_rmin = args_group['filter_rmin'] if filter_type != 'None' else None
volume_constraint = VolumeConstraint(mesh=mesh, volfrac=volfrac,
                                    filter_type=filter_type,
                                    filter_rmin=filter_rmin) 

space_degree = args.degree
solver_method = args.solver_method
compliance_objective = ComplianceObjective(
    mesh=mesh,
    space_degree=space_degree,
    dof_per_node=2,
    dof_ordering='gd-priority', 
    material_properties=material_properties,
    filter_type=filter_type,
    filter_rmin=filter_rmin,
    pde=pde,
    solver_method=solver_method, 
    volume_constraint=volume_constraint
)

options = opt_alg_options(
    x0=material_properties.rho,
    objective=compliance_objective,
    MaxIters=500,
    FunValDiff=0.01
)

oc_optimizer = OCAlg(options)
oc_optimizer.run()