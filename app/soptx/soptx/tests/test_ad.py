"""Test cases for compliance objective and volume constraint."""

from dataclasses import dataclass
import dataclasses
from typing import Literal, Optional, Union, Dict, Any

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d, UniformMesh3d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
    ElasticMaterialConfig,
    ElasticMaterialProperties,
    SIMPInterpolation
)
from soptx.pde import MBBBeam2dData1, Cantilever2dData1, Cantilever3dData1
from soptx.solver import ElasticFEMSolver, AssemblyMethod, AssemblyConfig
from soptx.opt import ComplianceObjective, VolumeConstraint
from soptx.filter import Filter, FilterConfig

bm.set_backend('pytorch')

mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=10, ny=10)
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
    projection_beta: Optional[float] = None  # Only for Heaviside projection
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
    elif config.problem_type == 'cantilever_3d':  
        pde = Cantilever3dData1(
            xmin=0, xmax=config.nx*h[0],
            ymin=0, ymax=config.ny*h[1],
            zmin=0, zmax=config.nz*h[2]
        )

    assembly_config = AssemblyConfig(
        method=config.assembly_method,
        quadrature_degree_increase=config.quadrature_degree_increase
    )

    # 设置默认的求解器参数
    default_solver_params = {
        'cg': {'maxiter': 5000, 'atol': 1e-12, 'rtol': 1e-12},
        'direct': {'solver_type': 'mumps'}
    }
    solver_params = config.solver_params or default_solver_params[config.solver_type]

    solver = ElasticFEMSolver(
       material_properties=material_properties,
       tensor_space=tensor_space_C,
       pde=pde,
       assembly_config=assembly_config,
       solver_type=config.solver_type,  # 添加默认求解器类型
       solver_params=solver_params  # 添加求解器参数
   )
    
    # Initialize density field
    array = config.volume_fraction * bm.ones(mesh.number_of_cells(), dtype=bm.float64)
    rho = space_D.function(array)
    
    return mesh, space_D, material_properties, solver, rho
    

def test_compliance_objective(config: TestConfig):
    """Test compliance objective computation and sensitivity analysis."""
    print(f"\n=== Testing Compliance Objective with {config.filter_type} filter ===")
    
    # Create base components
    mesh, space_D, material_properties, solver, rho = create_base_components(config)
    
    # Test solver
    solver.update_density(rho[:])
    solver_result = solver.solve_cg()
    displacement = solver_result.displacement
    print(f"\nSolver information:")
    print(f"- Displacement shape: {displacement.shape}:\n {displacement[:]}")
    
    # Create filter
    filter_config = FilterConfig(
        filter_type={'sensitivity': 0, 'density': 1, 'heaviside': 2}[config.filter_type],
        filter_radius=config.filter_radius
    )
    filter_obj = Filter(filter_config)
    filter_obj.initialize(mesh)
    
    # Create objective and constraint
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
    
    # Test objective function
    obj_value = objective.fun(rho=rho[:], u=displacement)
    print(f"Objective function value: {obj_value:.6e}")
    
    # Test element compliance
    ce = objective.get_element_compliance()
    print(f"\nElement compliance information:")
    print(f"- Shape: {ce.shape}:\n {ce}")
    print(f"- Min: {bm.min(ce):.6e}")
    print(f"- Max: {bm.max(ce):.6e}")
    print(f"- Mean: {bm.mean(ce):.6e}")
    
    # Test sensitivity
    dce = objective.jac(rho=rho[:], u=displacement, diff_mode="manual")
    print(f"\nElement sensitivity information (manual):")
    print(f"- Shape: {dce.shape}:\n, {dce}")
    print(f"- Min: {bm.min(dce):.6e}")
    print(f"- Max: {bm.max(dce):.6e}")
    print(f"- Mean: {bm.mean(dce):.6e}")

    dce_auto = objective.jac(rho=rho[:], u=displacement, diff_mode='auto')
    print(f"\nElement sensitivity information (auto):")
    print(f"- Shape: {dce_auto.shape}:\n, {dce_auto}")
    print(f"- Min: {bm.min(dce_auto):.6e}")
    print(f"- Max: {bm.max(dce_auto):.6e}")
    print(f"- Mean: {bm.mean(dce_auto):.6e}")
    
    # Test constraint
    constraint_value = constraint.fun(rho=rho[:])
    print(f"\nConstraint function value: {constraint_value:.6e}")
    
    gradient = constraint.jac(rho=rho[:])
    print(f"\nConstraint gradient information:")
    print(f"- Shape: {gradient.shape}")
    print(f"- Min: {bm.min(gradient):.6e}")
    print(f"- Max: {bm.max(gradient):.6e}")
    print(f"- Mean: {bm.mean(gradient):.6e}")
    
    return {
        'objective_value': obj_value,
        'element_compliance': ce,
        'sensitivity': dce,
        'constraint_value': constraint_value,
        'constraint_gradient': gradient
    }

if __name__ == "__main__":
    # Test 3D case with density filter
    config_3d = TestConfig(
        problem_type='cantilever_3d',
        nx=60, ny=20, nz=4,
        volume_fraction=0.3,
        filter_radius=1.5,
        filter_type='density'
    )
    # results_3d = test_compliance_objective(config_3d)
    
    # Test 2D case with sensitivity filter
    config_2d = TestConfig(
        problem_type='mbb_2d',
        nx=60, ny=20,
        volume_fraction=0.5,
        filter_radius=2.4,
        filter_type='sensitivity'
    )
    # results_2d = test_compliance_objective(config_2d)

    # Test 2D case with sensitivity filter
    config_cantilever_2d = TestConfig(
        problem_type='cantilever_2d',
        nx=160, ny=100,
        # nx=8, ny=5,
        volume_fraction=0.4,
        filter_radius=6,
        filter_type='sensitivity'
    )

    result = test_compliance_objective(config_cantilever_2d)