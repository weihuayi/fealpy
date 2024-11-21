"""
测试 elastic_fem_solver.py 的功能
"""
from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
    ElasticMaterialConfig,
    ElasticMaterialProperties,
    SIMPInterpolation
)
from soptx.solver import ElasticFEMSolver

from soptx.pde import MBBBeam2dData1


def test_elastic_fem_solver():
    print("\n=== 弹性有限元求解测试 ===")
    
    # 1. 创建网格
    nx, ny = 2, 2
    extent = [0, nx, 0, ny]
    h = [1.0, 1.0]
    origin = [0.0, 0.0]
    mesh = UniformMesh2d(
                        extent=extent, h=h, origin=origin,
                        ipoints_ordering='yx', flip_direction='y',
                        device='cpu'
                    )
    
    # 2. 创建函数空间
    p = 1
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, 2))
    space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
    
    # 3. 创建材料属性
    # 材料配置
    material_config = ElasticMaterialConfig(
                                            elastic_modulus=1.0,
                                            minimal_modulus=1e-9,
                                            poisson_ratio=0.3,
                                            plane_assumption="plane_stress"
                                        )
    
    # 插值模型
    interpolation_model = SIMPInterpolation(penalty_factor=3.0)
    
    # 密度场（初始为均匀分布）
    volfrac = 0.5
    array = volfrac * bm.ones(mesh.number_of_cells(), dtype=bm.float64)
    rho_elem = space_D.function(array)
    
    # 创建材料属性对象
    material_properties = ElasticMaterialProperties(
                                config=material_config,
                                rho=rho_elem[:],
                                interpolation_model=interpolation_model
                            )
    
    # 4. 创建PDE问题
    pde = MBBBeam2dData1(xmin=0, xmax=nx*h[0], ymin=0, ymax=ny*h[1])
    
    # 5. 创建求解器
    solver = ElasticFEMSolver(
                material_properties=material_properties,
                tensor_space=tensor_space_C,
                pde=pde
            )

    # 7. 测试基础刚度矩阵
    # 测试基础局部刚度矩阵
    base_local_K = solver.get_base_local_stiffness_matrix()
    print("\n=== 基础局部刚度矩阵信息 ===")
    print(f"基础局部刚度矩阵 - {base_local_K.shape}:\n {base_local_K[0]}")

    # 测试局部刚度矩阵计算
    local_K = solver.compute_local_stiffness_matrix()
    print("\n=== 当前材料局部刚度矩阵信息 ===")
    print(f"局部刚度矩阵 - {local_K.shape}:\n {local_K[0]}")

    print("\n=== 测试共轭梯度求解器 ===")
    try:
        cg_result = solver.solve_cg()
        
        print(f"CG求解结果:")
        # print(f"- 迭代次数: {cg_result.iterations}")
        # print(f"- 是否收敛: {cg_result.converged}")
        # print(f"- 残差: {cg_result.residual}")
        print(f"- 位移场形状: {cg_result.displacement.shape}")
        print(f"- 最大位移: {bm.max(bm.abs(cg_result.displacement))}")

        # 获取求解后的全局矩阵和向量
        K = solver.get_global_stiffness_matrix()
        F = solver.get_global_force_vector()
        if K is not None and F is not None:
            print("\n=== 全局矩阵和载荷向量信息 ===")
            print(f"全局刚度矩阵 - {K.shape}:\n {K.to_dense().round(4)}")
            print(f"全局载荷向量 -  {F.shape}:\n {F[:]}")
            print(f"载荷向量最大值: {bm.max(bm.abs(F))}")
    except Exception as e:
        print(f"CG求解失败: {str(e)}")

    print("\n=== 测试直接求解器 ===")
    try:
        direct_result = solver.solve_direct(solver_type='mumps')
        print(f"直接求解结果:")
        print(f"- 位移场形状: {direct_result.displacement.shape}")
        print(f"- 最大位移: {bm.max(bm.abs(direct_result.displacement))}")
        
        # 获取求解后的全局矩阵和向量
        K = solver.get_global_stiffness_matrix()
        F = solver.get_global_force_vector()
        if K is not None and F is not None:
            print("\n=== 全局矩阵和载荷向量信息 ===")
            print(f"全局刚度矩阵 - {K.shape}:\n {K.to_dense().round(4)}")
            print(f"全局载荷向量 -  {F.shape}:\n {F[:]}")
            print(f"载荷向量最大值: {bm.max(bm.abs(F))}")
    except Exception as e:
        print(f"直接求解失败: {str(e)}")

if __name__ == "__main__":
    test_elastic_fem_solver()