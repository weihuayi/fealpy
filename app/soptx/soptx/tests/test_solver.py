"""测试 solver 模块求解位移."""

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
from soptx.solver import ElasticFEMSolver, AssemblyMethod

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    # 问题参数
    problem_type: Literal['mbb_2d', 'cantilever_2d', 'cantilever_3d']
    nx: int
    ny: int
    nz: int
    volume_fraction: float
    penalty_factor: float = 3.0
    mesh_type: Literal['uniform_mesh_2d', 'triangle_mesh']
    
    # 材料参数
    elastic_modulus: float = 1.0
    poisson_ratio: float = 0.3
    minimal_modulus: float = 1e-9
    
    # 求解参数
    assembly_method: AssemblyMethod                 # 矩阵组装方法
    solver_type: Literal['cg', 'direct'] = 'direct' # 求解器类型
    solver_params: Optional[Dict[str, Any]] = None  # 求解器参数

    # 其它参数
    save_dir: Union[str, Path]


def create_base_components(config: TestConfig):
    """Create basic components needed for topology optimization based on configuration."""
    # Create mesh
    if config.problem_type == 'cantilever_3d':
        extent = [0, config.nx, 0, config.ny, 0, config.nz]
        h = [1.0, 1.0, 1.0]
        origin = [0.0, 0.0, 0.0]
        mesh = UniformMesh3d(
                    extent=extent, h=h, origin=origin,
                    ipoints_ordering='zyx', flip_direction='y',
                    device='cpu'
                )
    else:
        extent = [0, config.nx, 0, config.ny]
        h = [1.0, 1.0]
        origin = [0.0, 0.0]
        mesh = UniformMesh2d(
                    extent=extent, h=h, origin=origin,
                    ipoints_ordering='yx', flip_direction='y',
                    device='cpu'
                )
    GD = mesh.geo_dimension()
    
    # Create function spaces
    p = 1
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, GD))
    space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
    
    # Create material properties
    material_config = ElasticMaterialConfig(
            elastic_modulus=config.elastic_modulus,
            minimal_modulus=config.minimal_modulus,
            poisson_ratio=config.poisson_ratio,
            plane_assumption="3D" if GD == 3 else "plane_stress"
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

    solver = ElasticFEMSolver(
                material_properties=material_properties,
                tensor_space=tensor_space_C,
                pde=pde,
                assembly_method=config.assembly_method,
                solver_type=config.solver_type,
                solver_params=config.solver_params 
            )
    
    array = config.volume_fraction * bm.ones(mesh.number_of_cells(), dtype=bm.float64)
    rho = space_D.function(array)
    
    return mesh, space_D, material_properties, solver, rho

def run_solver_test(config: TestConfig):
    """测试不同矩阵组装方法的 solver."""
    print(f"\n=== Testing Solver with {config.assembly_method} ===")
    
    # Create base components
    mesh, space_D, material_properties, solver, rho = create_base_components(config)
    
    # Test solver
    solver.update_density(rho[:])
    solver_result = solver.solve()
    displacement = solver_result.displacement
    print(f"\nSolver information:")
    print(f"- Displacement shape: {displacement.shape}:\n {displacement[:]}")

    mesh.nodedata["uh"] = displacement
    base_dir = '/home/heliang/FEALPy_Development/fealpy/app/soptx/soptx/vtu'
    mesh.to_vtk(f'{base_dir}/uh_3d.vtu')

    # base_local_K = solver.get_base_local_stiffness_matrix()
    # print("\n=== 基础局部刚度矩阵信息 ===")
    # print(f"基础局部刚度矩阵 - {base_local_K.shape}:\n {base_local_K[0]}")
    # local_K = solver.compute_local_stiffness_matrix()
    # print("\n=== 当前材料局部刚度矩阵 ===")
    # print(f"局部刚度矩阵 - {local_K.shape}:\n {local_K.round(4)}")

    # K = solver.get_global_stiffness_matrix()
    # F = solver.get_global_force_vector()
    # print("\n=== 全局矩阵和载荷向量信息 ===")
    # print(f"全局刚度矩阵 - {K.shape}:\n {K.to_dense().round(4)}")
    # print(f"全局刚度矩阵最大值: {bm.max(bm.abs(K.to_dense()))}")
    # print(f"全局载荷向量 -  {F.shape}:\n {F[:]}")

if __name__ == "__main__":
    config1 = TestConfig(
                problem_type='cantilever_3d',
                nx=60, ny=20, nz=4,
                volume_fraction=0.3,
                penalty_factor=3.0,
                mesh_type='uniform_mesh_3d',
                assembly_method=AssemblyMethod.FAST_3D_UNIFORM,
                solver_type='direct',
                solver_params={'solver_type': 'mumps'}
            )
    config2 = TestConfig(
                problem_type='cantilever_3d',
                nx=60, ny=20, nz=4,
                volume_fraction=0.3,
                penalty_factor=3.0,
                mesh_type='uniform_mesh_3d',
                assembly_method=AssemblyMethod.FAST_3D_UNIFORM,
                solver_type='cg',
                solver_params={'maxiter': 1000, 'atol': 1e-8, 'rtol': 1e-8}   
            )
    result = run_solver_test(config1)