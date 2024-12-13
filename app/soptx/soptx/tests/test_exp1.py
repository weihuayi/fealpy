
from fealpy.mesh import UniformMesh2d, UniformMesh3d, TriangleMesh

from dataclasses import dataclass
import dataclasses
from typing import Literal, Optional, Union, Dict, Any
from soptx.solver import (ElasticFEMSolver, 
                          AssemblyMethod, 
                          AssemblyConfig)
from soptx.filter import Filter, FilterConfig
from soptx.opt import ComplianceObjective, VolumeConstraint
from soptx.opt import OCOptimizer, save_optimization_history

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    # Required parameters (no default values)
    problem_type: Literal['mbb_2d', 'cantilever_2d', 'cantilever_3d']
    nx: int
    ny: int
    volume_fraction: float
    filter_radius: float
    filter_type: Literal['sensitivity', 'density', 'heaviside']
    
    # Optional parameters (with default values)
    nz: Optional[int] = None  # Only for 3D problems
    elastic_modulus: float = 1.0
    poisson_ratio: float = 0.3
    minimal_modulus: float = 1e-9
    penalty_factor: float = 3.0
    projection_beta: Optional[float] = None

    # OC yo
    max_iterations: int = 200
    move_limit: float = 0.2
    tolerance: float = 0.01
    initial_lambda: float = 1e9
    bisection_tol: float = 1e-3

    assembly_method: AssemblyMethod = AssemblyMethod.STANDARD  # 矩阵组装方法
    quadrature_degree_increase: int = 3  # 积分阶数增量
    solver_type: Literal['cg', 'direct'] = 'cg'  # 求解器类型
    solver_params: Optional[Dict[str, Any]] = None  # 求解器参数

def create_base_components(config: TestConfig):
    """Create basic components needed for topology optimization based on configuration."""
    # Create mesh and determine dimensionality
    if config.problem_type == 'cantilever_3d':
        extent = [0, config.nx, 0, config.ny, 0, config.nz]
        h = [1.0, 1.0, 1.0]
        origin = [0.0, 0.0, 0.0]
        mesh = UniformMesh3d(
            extent=extent, h=h, origin=origin,
            ipoints_ordering='zyx', flip_direction='y',
            device='cpu'
        )
        dimension = 3
    else:
        extent = [0, config.nx, 0, config.ny]
        h = [1.0, 1.0]
        origin = [0.0, 0.0]
        # mesh = UniformMesh2d(
        #     extent=extent, h=h, origin=origin,
        #     ipoints_ordering='yx', flip_direction='y',
        #     device='cpu'
        # )
        mesh = TriangleMesh.from_box(box=[0, config.nx*h[0], 0, config.ny*h[1]], 
                                    nx=config.nx, ny=config.ny, device='cpu')
        dimension = 2

def run_optimization_test(config: TestConfig) -> Dict[str, Any]:
    """Run topology optimization test with given configuration."""
    # Create base components
    mesh, space_D, material_properties, solver, rho = create_base_components(config)
    
    # Create filter based on configuration
    filter_config = FilterConfig(
                        filter_type={'sensitivity': 0, 
                                     'density': 1, 
                                     'heaviside': 2}[config.filter_type],
                        filter_radius=config.filter_radius
                    )
    filter_obj = Filter(filter_config)
    filter_obj.initialize(mesh)
    
    # Create optimization components
    objective = ComplianceObjective(
                    material_properties=material_properties,
                    solver=solver,
                    filter=filter_obj
                )
    constraint = VolumeConstraint(
                    mesh=mesh,
                    volume_fraction=config.volume_fraction,
                    filter=filter_obj
                )
    
    # Create optimizer
    optimizer = OCOptimizer(
                    objective=objective,
                    constraint=constraint,
                    filter=filter_obj,
                    options={
                        'max_iterations': config.max_iterations,
                        'move_limit': config.move_limit,
                        'tolerance': config.tolerance,
                        'initial_lambda': config.initial_lambda,
                        'bisection_tol': config.bisection_tol
                    }
                )
    
    # Prepare optimization parameters
    opt_params = {}
    if config.filter_type == 'heaviside' and config.projection_beta is not None:
        opt_params['beta'] = config.projection_beta
    
    # Run optimization
    rho_opt, history = optimizer.optimize(rho=rho[:], **opt_params)
    
    # Save results
    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    save_optimization_history(mesh, history, str(save_path))
    
    return {
        'optimal_density': rho_opt,
        'history': history,
        'mesh': mesh
    }