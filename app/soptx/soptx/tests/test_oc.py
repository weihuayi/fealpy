"""测试 OC 优化算法的功能"""

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from soptx.material import (
    ElasticMaterialConfig,
    ElasticMaterialProperties,
    SIMPInterpolation
)
from soptx.pde import MBBBeam2dData1
from soptx.solver import ElasticFEMSolver
from soptx.opt import ComplianceObjective, VolumeConstraint
from soptx.filter import Filter, FilterConfig
from soptx.opt import OCOptimizer

def test_oc_optimizer():
    """测试 OC 优化器的主要功能"""
    
    print("\n=== OC 优化器测试 ===")
    
    #---------------------------------------------------------------------------
    # 1. 创建计算网格和函数空间
    #---------------------------------------------------------------------------
    # 创建网格
    nx, ny = 60, 20
    extent = [0, nx, 0, ny]
    h = [1.0, 1.0]
    origin = [0.0, 0.0]
    mesh = UniformMesh2d(
        extent=extent, h=h, origin=origin,
        ipoints_ordering='yx', flip_direction='y',
        device='cpu'
    )
    
    # 创建函数空间
    p = 1
    space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
    tensor_space_C = TensorFunctionSpace(space_C, (-1, 2))
    space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
    
    #---------------------------------------------------------------------------
    # 2. 创建材料属性和PDE问题
    #---------------------------------------------------------------------------
    # 创建材料配置和插值模型
    material_config = ElasticMaterialConfig(
        elastic_modulus=1.0,
        minimal_modulus=1e-9,
        poisson_ratio=0.3,
        plane_assumption="plane_stress"
    )
    interpolation_model = SIMPInterpolation(penalty_factor=3.0)
    
    # 创建材料属性计算器
    material_properties = ElasticMaterialProperties(
        config=material_config,
        interpolation_model=interpolation_model
    )

    # 创建 PDE 问题
    pde = MBBBeam2dData1(xmin=0, xmax=nx*h[0], ymin=0, ymax=ny*h[1])
    
    #---------------------------------------------------------------------------
    # 3. 创建求解器
    #---------------------------------------------------------------------------
    solver = ElasticFEMSolver(
        material_properties=material_properties,
        tensor_space=tensor_space_C,
        pde=pde
    )

    # 初始密度场
    volfrac = 0.5
    array = volfrac * bm.ones(mesh.number_of_cells(), dtype=bm.float64)
    rho = space_D.function(array)

    # 更新求解器中的密度
    solver.update_density(rho[:])

    # 检查基础刚度矩阵
    ke0 = solver.get_base_local_stiffness_matrix()
    print(f"\n基础材料局部刚度矩阵 - {ke0.shape}:")
    print(f"{ke0[0]}")

    #---------------------------------------------------------------------------
    # 4. 创建滤波器和优化组件
    #---------------------------------------------------------------------------
    # 创建灵敏度滤波器
    filter_sens = Filter(FilterConfig(filter_type=1, filter_radius=2.4))
    filter_sens.initialize(mesh)

    # 创建目标函数
    objective = ComplianceObjective(
        material_properties=material_properties,
        solver=solver,
        filter=filter_sens
    )

    # 创建体积约束
    constraint = VolumeConstraint(
        mesh=mesh,
        volume_fraction=volfrac,
        filter=filter_sens
    )

    # 创建优化器
    oc_optimizer = OCOptimizer(
        objective=objective,
        constraint=constraint,
        filter=filter_sens,
        options={
            'max_iterations': 200,
            'move_limit': 0.2,
            'tolerance': 0.01,
            'initial_lambda': 1e9,
            'bisection_tol': 1e-3
        }
    )

    #---------------------------------------------------------------------------
    # 5. 测试优化过程
    #---------------------------------------------------------------------------
    print("\n=== 开始优化测试 ===")
    try:
        # # 计算初始状态
        # u = solver.solve_cg().displacement

        # # 测试目标函数及其导数
        # dc = objective.jac(rho=rho[:], u=u)
        # print(f"\n单元灵敏度值 - {dc.shape}:\n {dc}")
        
        # obj_val = objective.fun(rho=rho[:], u=u)
        # print(f"\n目标函数值: {obj_val:.6e}")

        # # 测试约束函数及其导数
        # con_val = constraint.fun(rho=rho[:])
        # print(f"约束函数值: {con_val:.6e}")
        
        # con_grad = constraint.jac(rho=rho[:])
        # print(f"约束函数梯度 - {con_grad.shape}:\n {con_grad}")
        
        # 运行优化
        print("\n开始优化迭代...")
        rho_opt = oc_optimizer.optimize(rho=rho[:])
        
        # 输出优化结果统计
        print("\n优化结果统计:")
        print(f"- 最终密度均值: {bm.mean(rho_opt):.3f}")
        print(f"- 最终密度最大值: {bm.max(rho_opt):.3f}")
        print(f"- 最终密度最小值: {bm.min(rho_opt):.3f}")
        
    except Exception as e:
        print(f"\n优化过程失败: {str(e)}")
        raise e

if __name__ == "__main__":
    test_oc_optimizer()