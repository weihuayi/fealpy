"""测试 solver 模块求解位移."""

from dataclasses import dataclass
from typing import Literal, Dict, Any

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
                            ElasticMaterialConfig,
                            ElasticMaterialInstance,
                        )
from soptx.pde import Cantilever2dData1, Cantilever2dData2
from soptx.solver import ElasticFEMSolver, AssemblyMethod
from soptx.utils import timer

@dataclass
class TestConfig:
    """Configuration for topology optimization test cases."""
    backend: Literal['numpy', 'pytorch']
    pde_type: Literal['cantilever_2d_1', 'cantilever_2d_2']

    elastic_modulus: float
    poisson_ratio: float
    minimal_modulus: float

    domain_length : float
    domain_width : float

    load : float

    volume_fraction: float
    penalty_factor: float

    mesh_type: Literal['uniform_mesh_2d', 'triangle_mesh']
    nx: int
    ny: int
    
    assembly_method: AssemblyMethod
    solver_type: Literal['cg', 'direct'] 
    solver_params: Dict[str, Any] 

def create_base_components(config: TestConfig):
    """Create basic components needed for topology optimization based on configuration."""
    if config.backend == 'numpy':
        bm.set_backend('numpy')
    elif config.backend == 'pytorch':
        bm.set_backend('pytorch')

    if config.pde_type == 'cantilever_2d_2':
        pde = Cantilever2dData2(
                    xmin=0, xmax=config.domain_length,
                    ymin=0, ymax=config.domain_width,
                    T = config.load
                )
        if config.mesh_type == 'triangle_mesh':
            mesh = TriangleMesh.from_box(box=pde.domain(), nx=config.nx, ny=config.ny)
    elif config.pde_type == 'cantilever_2d_1':
        pde = Cantilever2dData1(
                    xmin=0, xmax=extent[1] * h[0],
                    ymin=0, ymax=extent[3] * h[1]
                )
        if config.mesh_type == 'uniform_mesh_2d':
            extent = [0, config.nx, 0, config.ny]
            h = [1.0, 1.0]
            origin = [0.0, 0.0]
            mesh = UniformMesh2d(
                        extent=extent, h=h, origin=origin,
                        ipoints_ordering='yx', flip_direction=None,
                        device='cpu'
                    )

    GD = mesh.geo_dimension()
    
    p = 3
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, GD))
    space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
    
    material_config = ElasticMaterialConfig(
                            elastic_modulus=config.elastic_modulus,            
                            minimal_modulus=config.minimal_modulus,         
                            poisson_ratio=config.poisson_ratio,            
                            plane_assumption="plane_stress",    
                            interpolation_model="SIMP",    
                            penalty_factor=config.penalty_factor
                        )
    
    materials = ElasticMaterialInstance(config=material_config)
    
    array = config.volume_fraction * bm.ones(mesh.number_of_cells(), dtype=bm.float64)
    rho = space_D.function(array)
    
    return materials, tensor_space_C, pde, rho

def run_assmeble_test(config: TestConfig):
    """测试 SOPTX 中不同的 assembly_method."""
    materials, tensor_space_C, pde, rho = create_base_components(config)

    solver = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=config.assembly_method,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    for i in range(5):
        # 创建计时器
        t = timer(f"{config.assembly_method} Timing")
        next(t)  # 启动计时器
        solver.update_status(rho[:])
        t.send('准备时间')
        K = solver._assemble_global_stiffness_matrix()
        t.send('组装时间')
        t.send(None)

def run_solve_test(config: TestConfig):
    """测试 SOPTX 中不同的 solver_type."""
    materials, tensor_space_C, pde, rho = create_base_components(config)

    solver = ElasticFEMSolver(
                    materials=materials,
                    tensor_space=tensor_space_C,
                    pde=pde,
                    assembly_method=config.assembly_method,
                    solver_type=config.solver_type,
                    solver_params=config.solver_params 
                )
    for i in range(3):
        # 创建计时器
        t = timer(f"{config.solver_type} Timing")
        next(t)  # 启动计时器
        solver.update_status(rho[:])
        t.send('准备时间')
        solver_result = solver.solve()
        uh = solver_result.displacement
        t.send('求解时间')
        t.send(None)


if __name__ == "__main__":
    config_standard_assemble = TestConfig(
                                    backend='numpy',
                                    pde_type='cantilever_2d_2',
                                    elastic_modulus=1e5, poisson_ratio=0.3, minimal_modulus=1e-9,
                                    domain_length=3.0, domain_width=1.0,
                                    load=2000,
                                    volume_fraction=0.5,
                                    penalty_factor=3.0,
                                    mesh_type='triangle_mesh', nx=300, ny=100,
                                    assembly_method=AssemblyMethod.STANDARD,
                                    solver_type='direct', solver_params={'solver_type': 'mumps'},
                                )
    config_symbolic_assemble = TestConfig(
                                    backend='numpy',
                                    pde_type='cantilever_2d_2',
                                    elastic_modulus=1e5, poisson_ratio=0.3, minimal_modulus=1e-9,
                                    domain_length=3.0, domain_width=1.0,
                                    load=2000,
                                    volume_fraction=0.5,
                                    penalty_factor=3.0,
                                    mesh_type='triangle_mesh', nx=300, ny=100,
                                    assembly_method=AssemblyMethod.SYMBOLIC,
                                    solver_type='direct', solver_params={'solver_type': 'mumps'},
                                )
    config_cg_solve = TestConfig(
                            backend='numpy',
                            pde_type='cantilever_2d_2',
                            elastic_modulus=1e5, poisson_ratio=0.3, minimal_modulus=1e-9,
                            domain_length=3.0, domain_width=1.0,
                            load=2000,
                            volume_fraction=0.5,
                            penalty_factor=3.0,
                            mesh_type='triangle_mesh', nx=300, ny=100,
                            assembly_method=AssemblyMethod.STANDARD,
                            solver_type='cg',  
                            solver_params={'maxiter': 1000, 'atol': 1e-8, 'rtol': 1e-8}  
                        )
    config_mumps_solve = TestConfig(
                                backend='numpy',
                                pde_type='cantilever_2d_2',
                                elastic_modulus=1e5, poisson_ratio=0.3, minimal_modulus=1e-9,
                                domain_length=3.0, domain_width=1.0,
                                load=2000,
                                volume_fraction=0.5,
                                penalty_factor=3.0,
                                mesh_type='triangle_mesh', nx=300, ny=100,
                                assembly_method=AssemblyMethod.STANDARD,
                                solver_type='direct', 
                                solver_params={'solver_type': 'mumps'},
                            )
    # result1 = run_assmeble_test(config_standard_assemble)
    result2 = run_assmeble_test(config_symbolic_assemble)