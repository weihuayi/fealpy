import argparse

from fealpy.backend import backend_manager as bm

from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
from fealpy.functionspace.lagrange_fe_space import LagrangeFESpace
from fealpy.functionspace.tensor_space import TensorFunctionSpace

from fealpy.opt import opt_alg_options

from soptx.pde import MBBBeam2dData1
from soptx.material import (
    ElasticMaterialConfig,
    ElasticMaterialProperties,
    SIMPInterpolation
)
from soptx.solver import FEMSolver
from soptx.filter import create_filter_properties, TopologyFilter

from app.soptx.soptx.opt.volume_objective import VolumeConstraint
from app.soptx.soptx.opt.compliance_objective import ComplianceObjective
from app.soptx.soptx.opt.oc_alg import OCAlg

parameter_groups = {
    'group1': {'nx': 60, 'ny': 20, 'filter_rmin': 2.4},
    'group2': {'nx': 150, 'ny': 50, 'filter_rmin': 6.0},
    'group3': {'nx': 300, 'ny': 100, 'filter_rmin': 16.0},
    'group4': {'nx': 300, 'ny': 100, 'filter_rmin': 9},
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compliance minimization on MBB beam.")

    parser.add_argument('--backend', 
                        default='numpy', type=str,
                        help='Specify the backend type for computation, default is "numpy".')

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

    return parser.parse_args()

def setup_problem(args):
    bm.set_backend(args.backend)
    args_group = parameter_groups[args.group]

    nx, ny = args_group['nx'], args_group['ny']
    pde = MBBBeam2dData1()

    extent = [0, nx, 0, ny]
    h = [1.0, 1.0]
    origin = [0.0, 0.0]
    mesh = UniformMesh2d(extent=extent, h=h, origin=origin, 
                        ipoints_ordering='yx', flip_direction='y', 
                        device='cpu')
    
    p = args.degree
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, 2))

    volfrac = args.volfrac
    rho = volfrac * bm.ones(nx * ny, dtype=bm.float64)

    # 创建材料配置对象
    material_config = ElasticMaterialConfig(
                                            elastic_modulus=1.0,
                                            minimal_modulus=1e-9,
                                            poisson_ratio=0.3,
                                            plane_assumption="plane_stress"
                                        )

    # 创建插值模型实例
    interpolation_model = SIMPInterpolation(penalty_factor=3.0)

    # 创建材料属性对象
    material_properties = ElasticMaterialProperties(
                                                    config=material_config,
                                                    rho=rho,
                                                    interpolation_model=interpolation_model
                                                )

    fem_solver = FEMSolver(
                            material_properties=material_properties,
                            tensor_space=tensor_space_C,
                            pde=pde
                        )
    solver_method = args.solver_method
    uh = fem_solver.solve(solver_method=solver_method)
    



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
    return oc_optimizer

def main():
    args = parse_arguments()
    oc_optimizer = setup_problem(args)
    oc_optimizer.run()

if __name__ == '__main__':
    main()