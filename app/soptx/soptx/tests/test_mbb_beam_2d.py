"""2D 悬臂梁"""

from dataclasses import dataclass
from typing import Literal, Optional, Union, Dict, Any
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
    ElasticMaterialConfig,
    ElasticMaterialProperties,
    SIMPInterpolation
)
from soptx.pde import MBBBeam2dData1
from soptx.solver import (ElasticFEMSolver, AssemblyMethod)
from soptx.filter import (Filter, FilterConfig)
from soptx.opt import ComplianceObjective, VolumeConstraint

from soptx.opt import OCOptimizer, save_optimization_history
from soptx.opt import MMAOptimizer

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    # Required parameters (no default values)
    nx: int
    ny: int
    volume_fraction: float
    filter_radius: float
    filter_type: Literal['sensitivity', 'density', 'heaviside']

    save_dir: Union[str, Path]
    mesh_type: Literal['uniform_mesh_2d', 'triangle_mesh']
    assembly_method: AssemblyMethod
    
    # Optional parameters (with default values)
    elastic_modulus: float = 1.0
    poisson_ratio: float = 0.3
    minimal_modulus: float = 1e-9
    penalty_factor: float = 3.0
    projection_beta: Optional[float] = None  # Only for Heaviside projection

    # 新增优化器类型参数
    optimizer_type: Literal['oc', 'mma'] = 'oc'  # 默认使用 OC 方法
    max_iterations: int = 100
    tolerance: float = 0.01
    
    # OC 优化器的参数
    move_limit: float = 0.2
    damping_coef: float = 0.5
    initial_lambda: float = 1e9
    bisection_tol: float = 1e-3

    # MMA 优化器的参数
    asymp_init: float = 0.5
    asymp_incr: float = 1.2
    asymp_decr: float = 0.7


def create_base_components(config: TestConfig):
    """Create basic components needed for topology optimization based on configuration."""
    extent = [0, config.nx, 0, config.ny]
    h = [1.0, 1.0]
    if config.mesh_type == 'uniform_mesh_2d':
        origin = [0.0, 0.0]
        mesh = UniformMesh2d(
                    extent=extent, h=h, origin=origin,
                    ipoints_ordering='yx', flip_direction=None,
                    device='cpu'
                )
    elif config.mesh_type == "triangle_mesh":
        mesh = TriangleMesh.from_box(box=[0, config.nx*h[0], 0, config.ny*h[1]], 
                                    nx=config.nx, ny=config.ny, device='cpu')
    
    p = 1
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, 2))
    print(f"gdof:", {tensor_space_C.number_of_global_dofs()})
    space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
    
    # Create material properties
    material_config = ElasticMaterialConfig(
                            elastic_modulus=config.elastic_modulus,
                            minimal_modulus=config.minimal_modulus,
                            poisson_ratio=config.poisson_ratio,
                            plane_assumption="plane_stress"
                        )
    interpolation_model = SIMPInterpolation(penalty_factor=config.penalty_factor)
    material_properties = ElasticMaterialProperties(
                                config=material_config,
                                interpolation_model=interpolation_model
                            )
    
    pde = MBBBeam2dData1(
                xmin=0, xmax=extent[1] * h[0],
                ymin=0, ymax=extent[3] * h[1]
            )
    
    solver = ElasticFEMSolver(
                    material_properties=material_properties,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=config.assembly_method,
                    solver_type = 'direct',  
                    solver_params = {'solver_type': 'mumps'} 
                    # solver_type='cg',  
                    # solver_params={'maxiter': 1000, 'atol': 1e-8, 'rtol': 1e-8}  
                )
    
    array = config.volume_fraction * bm.ones(mesh.number_of_cells(), dtype=bm.float64)
    rho = space_D.function(array)
    
    return mesh, space_D, material_properties, solver, rho

def run_optimization_test(config: TestConfig) -> Dict[str, Any]:
    """Run topology optimization test with given configuration."""
    # Create base components
    mesh, space_D, material_properties, solver, rho = create_base_components(config)
    NC = mesh.number_of_cells()
    
    # Create filter based on configuration
    filter_config = FilterConfig(
                    filter_type={'sensitivity': 0, 'density': 1, 'heaviside': 2}[config.filter_type],
                    filter_radius=config.filter_radius)
    filter_obj = Filter(filter_config)
    filter_obj.initialize(mesh)
    
    # Create optimization components
    objective = ComplianceObjective(
                    material_properties=material_properties,
                    solver=solver)

    constraint = VolumeConstraint(
                    mesh=mesh,
                    volume_fraction=config.volume_fraction)
    
    # 根据配置创建优化器
    if config.optimizer_type == 'oc':
        optimizer = OCOptimizer(
                        objective=objective,
                        constraint=constraint,
                        filter=filter_obj,
                        options={
                            'max_iterations': config.max_iterations,
                            'move_limit': config.move_limit,
                            'damping': config.damping_coef,
                            'tolerance': config.tolerance,
                            'initial_lambda': config.initial_lambda,
                            'bisection_tol': config.bisection_tol
                        }
                    )
    elif config.optimizer_type == 'mma':
        optimizer = MMAOptimizer(
                        objective=objective,
                        constraint=constraint,
                        filter=filter_obj,
                        options={
                            'max_iterations': config.max_iterations,
                            'tolerance': config.tolerance,
                            'm': 1,
                            'n': NC,
                            'xmin': bm.zeros(NC, dtype=bm.float64).reshape(-1, 1),
                            'xmax': bm.ones(NC, dtype=bm.float64).reshape(-1, 1),
                            "a0": 1,
                            "a": bm.zeros(1, dtype=bm.float64).reshape(-1, 1),
                            'c': 1e4 * bm.ones(1, dtype=bm.float64).reshape(-1, 1),
                            'd': bm.zeros(1, dtype=bm.float64).reshape(-1,),
                        }
                    )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")

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
    base_dir = '/home/heliang/FEALPy_Development/fealpy/app/soptx/soptx/vtu'

    '''
    参数来源论文: Efficient topology optimization in MATLAB using 88 lines of code
    '''
    filter_type1 = 'sensitivity'
    optimizer_type1 = 'oc'
    nx, ny = 60, 20
    NC = nx * ny
    config1 = TestConfig(
        nx=60, ny=20,
        volume_fraction=0.5,
        filter_radius=nx*0.04,
        filter_type=filter_type1,        # 指定使用灵敏度滤波器
        save_dir=f'{base_dir}/mbb_beam_2d_{optimizer_type1}_{filter_type1}_{NC}',
        mesh_type='uniform_mesh_2d',
        assembly_method=AssemblyMethod.FAST_STRESS_UNIFORM,
        optimizer_type=optimizer_type1,  # 指定使用 OC 优化器
        max_iterations=200,
        tolerance=0.01
    )

    '''
    参数来源论文: Efficient topology optimization in MATLAB using 88 lines of code
    '''
    nx, ny = 150, 50
    NC = nx * ny
    config2 = TestConfig(
        nx=nx, ny=ny,
        volume_fraction=0.5,
        filter_radius=nx*0.04,
        filter_type=filter_type1,        # 指定使用灵敏度滤波器
        save_dir=f'{base_dir}/mbb_beam_2d_{optimizer_type1}_{filter_type1}_{NC}',
        mesh_type='uniform_mesh_2d',
        assembly_method=AssemblyMethod.FAST_STRESS_UNIFORM,
        optimizer_type=optimizer_type1,  # 指定使用 OC 优化器
        max_iterations=200,
        tolerance=0.01
    )

    '''
    参数来源论文: Efficient topology optimization in MATLAB using 88 lines of code
    '''
    nx, ny = 300, 100
    NC = nx * ny
    config3 = TestConfig(
        nx=nx, ny=ny,
        volume_fraction=0.5,
        filter_radius=nx*0.04,
        filter_type=filter_type1,        # 指定使用灵敏度滤波器
        save_dir=f'{base_dir}/mbb_beam_2d_{optimizer_type1}_{filter_type1}_{NC}',
        mesh_type='uniform_mesh_2d',
        assembly_method=AssemblyMethod.FAST_STRESS_UNIFORM,
        optimizer_type=optimizer_type1,  # 指定使用 OC 优化器
        max_iterations=200,
        tolerance=0.01
    )

    '''
    参数来源论文: Efficient topology optimization in MATLAB using 88 lines of code
    '''
    filter_type2 = 'density'
    nx, ny = 60, 20
    NC = nx * ny
    config4 = TestConfig(
        nx=nx, ny=ny,
        volume_fraction=0.5,
        filter_radius=nx*0.04,
        filter_type=filter_type2,        # 指定使用灵敏度滤波器
        save_dir=f'{base_dir}/mbb_beam_2d_{optimizer_type1}_{filter_type2}_{NC}',
        mesh_type='uniform_mesh_2d',
        assembly_method=AssemblyMethod.FAST_STRESS_UNIFORM,
        optimizer_type=optimizer_type1,  # 指定使用 OC 优化器
        max_iterations=200,
        tolerance=0.01
    )

    result = run_optimization_test(config4)
    