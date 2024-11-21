"""Test cases for the OC optimization algorithm with configurable parameters and filtering schemes."""

from dataclasses import dataclass
from typing import Literal, Optional, Union, Dict, Any
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d, UniformMesh3d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
    ElasticMaterialConfig,
    ElasticMaterialProperties,
    SIMPInterpolation
)
from soptx.pde import MBBBeam2dData1, Cantilever2dData1, Cantilever3dData1
from soptx.solver import ElasticFEMSolver
from soptx.opt import ComplianceObjective, VolumeConstraint
from soptx.filter import Filter, FilterConfig
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
    max_iterations: int
    save_dir: Union[str, Path]
    
    # Optional parameters (with default values)
    nz: Optional[int] = None  # Only for 3D problems
    elastic_modulus: float = 1.0
    poisson_ratio: float = 0.3
    minimal_modulus: float = 1e-9
    penalty_factor: float = 3.0
    projection_beta: Optional[float] = None  # Only for Heaviside projection
    move_limit: float = 0.2
    tolerance: float = 0.01
    initial_lambda: float = 1e9
    bisection_tol: float = 1e-3
    

def create_base_components(config: TestConfig):
    """Create basic components needed for topology optimization based on configuration."""
    # Create mesh and determine dimensionality
    if config.problem_type == 'cantilever_3d':
        if config.nz is None:
            raise ValueError("nz must be specified for 3D problems")
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
        mesh = UniformMesh2d(
                        extent=extent, h=h, origin=origin,
                        ipoints_ordering='yx', flip_direction='y',
                        device='cpu'
                    )
        dimension = 2
    
    # Create function spaces
    p = 1
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, dimension))
    space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
    
    # Create material properties
    material_config = ElasticMaterialConfig(
                                    elastic_modulus=config.elastic_modulus,
                                    minimal_modulus=config.minimal_modulus,
                                    poisson_ratio=config.poisson_ratio,
                                    plane_assumption="3D" if dimension == 3 else "plane_stress"
                                )
    interpolation_model = SIMPInterpolation(penalty_factor=config.penalty_factor)
    material_properties = ElasticMaterialProperties(
                                            config=material_config,
                                            interpolation_model=interpolation_model
                                        )
    
    # Create PDE problem based on problem type
    if config.problem_type == 'mbb_2d':
        pde = MBBBeam2dData1(
                        xmin=0, xmax=config.nx*h[0],
                        ymin=0, ymax=config.ny*h[1]
                    )
    elif config.problem_type == 'cantilever_2d':
        pde = Cantilever2dData1(
                        xmin=0, xmax=config.nx*h[0],
                        ymin=0, ymax=config.ny*h[1]
                    )
    else:  # cantilever_3d
        pde = Cantilever3dData1(
                        xmin=0, xmax=config.nx*h[0],
                        ymin=0, ymax=config.ny*h[1],
                        zmin=0, zmax=config.nz*h[2]
                    )
    
    # Create solver
    # solver = ElasticFEMSolver(
    #                     material_properties=material_properties,
    #                     tensor_space=tensor_space_C,
    #                     pde=pde
    #                 )
    solver = ElasticFEMSolver(
       material_properties=material_properties,
       tensor_space=tensor_space_C,
       pde=pde,
       solver_type='cg',  # 添加默认求解器类型
       solver_params={'maxiter': 5000, 'atol': 1e-12, 'rtol': 1e-12}  # 添加求解器参数
   )
    
    # Initialize density field
    array = config.volume_fraction * bm.ones(mesh.number_of_cells(), dtype=bm.float64)
    rho = space_D.function(array)
    
    return mesh, space_D, material_properties, solver, rho

def run_optimization_test(config: TestConfig) -> Dict[str, Any]:
    """Run topology optimization test with given configuration."""
    # Create base components
    mesh, space_D, material_properties, solver, rho = create_base_components(config)
    
    # Create filter based on configuration
    filter_config = FilterConfig(
                            filter_type={'sensitivity': 0, 'density': 1, 'heaviside': 2}[config.filter_type],
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

if __name__ == "__main__":
    # Run single test
    config1 = TestConfig(
        problem_type='cantilever_2d',
        nx=160, ny=100,
        volume_fraction=0.4,
        filter_radius=6.0,
        filter_type='sensitivity',
        max_iterations=30,
        save_dir='/home/heliang/FEALPy_Development/fealpy/app/soptx/soptx/tests/cantilever_2d'
    )
    config2 = TestConfig(
        problem_type='mbb_2d',
        nx=60, ny=20,
        volume_fraction=0.5,
        filter_radius=2.4,
        filter_type='sensitivity',
        max_iterations=30,
        save_dir='/home/heliang/FEALPy_Development/fealpy/app/soptx/soptx/tests/mbb_2d'
    )
    result = run_optimization_test(config1)
    