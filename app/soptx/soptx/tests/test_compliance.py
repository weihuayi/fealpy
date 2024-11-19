"""
测试 compliance.py 中柔度目标函数的功能
"""
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
from soptx.opt import ComplianceObjective
from soptx.filter import Filter, FilterConfig

def test_compliance_objective():
    print("\n=== 柔度目标函数测试 ===")
    
    # 1. 创建网格
    nx, ny = 60, 20
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
        interpolation_model=interpolation_model
    )
    E = material_properties.material_model(rho=rho_elem[:])
    print(f"\n单元杨氏模量信息:")
    print(f"- 形状 - {E.shape}:\n {E}")
    print(f"杨氏模量均值: {bm.mean(E)}")
    print(f"杨氏模量最大值: {bm.max(E)}")
    print(f"杨氏模量最小值: {bm.min(E)}")
    
    # 4. 创建PDE问题
    pde = MBBBeam2dData1(xmin=0, xmax=nx*h[0], ymin=0, ymax=ny*h[1])
    
    # 5. 创建求解器
    solver = ElasticFEMSolver(
        material_properties=material_properties,
        tensor_space=tensor_space_C,
        pde=pde
    )

    # 6. 创建目标函数对象
    objective = ComplianceObjective(
        material_properties=material_properties,
        solver=solver
    )

    print("\n=== 测试目标函数计算 ===")
    try:
        # 求解位移场
        solver_result = solver.solve_cg()
        displacement = solver_result.displacement

        # 计算目标函数值
        obj_value = objective.fun(design_vars=rho_elem[:], 
                                state_vars=displacement)
        print(f"目标函数值: {obj_value:.6e}")

        # 获取单元柔顺度
        ce = objective.get_element_compliance()
        print(f"\n单元柔顺度信息:")
        print(f"- 形状 - {ce.shape}:\n {ce}")
        print(f"- 最小值: {bm.min(ce):.6e}")
        print(f"- 最大值: {bm.max(ce):.6e}")
        print(f"- 平均值: {bm.mean(ce):.6e}")

        # 测试不同密度下的目标函数值
        print("\n=== 测试不同密度下的目标函数值 ===")
        # 测试较小密度
        rho_small = 0.1 * bm.ones_like(rho_elem[:])
        obj_small = objective.fun(design_vars=rho_small[:], 
                                state_vars=displacement)
        print(f"密度=0.1时的目标函数值: {obj_small:.6e}")

        # 测试较大密度
        rho_large = 0.9 * bm.ones_like(rho_elem[:])
        obj_large = objective.fun(design_vars=rho_large[:],
                                state_vars=displacement)
        print(f"密度=0.9时的目标函数值: {obj_large:.6e}")

    except Exception as e:
        print(f"目标函数计算失败: {str(e)}")

    print("\n=== 测试目标函数梯度计算 ===")
    try:

        # 计算单元灵敏度（不使用滤波器）
        dce = objective.jac(design_vars=rho_elem[:], state_vars=displacement)
        print(f"\n原始单元灵敏度信息:")
        print(f"- 形状 - {dce.shape}:\n, {dce}")
        print(f"- 最小值: {bm.min(dce):.6e}")
        print(f"- 最大值: {bm.max(dce):.6e}")
        print(f"- 平均值: {bm.mean(dce):.6e}")

        # 测试不同类型的滤波器
        print("\n=== 测试不同类型的滤波器 ===")
        
        # 灵敏度滤波
        filter_sens = Filter(FilterConfig(filter_type=0, filter_radius=2.4))
        filter_sens.initialize(mesh)
        
        objective_sens = ComplianceObjective(
            material_properties=material_properties,
            solver=solver,
            filter=filter_sens
        )
        
        grad_filter_sens = objective_sens.jac(design_vars=rho_elem[:], state_vars=displacement)
        print(f"\n灵敏度滤波后的梯度信息:")
        print(f"- 形状 - {grad_filter_sens.shape}:\n, {grad_filter_sens}")
        print(f"- 最小值: {bm.min(grad_filter_sens):.6e}")
        print(f"- 最大值: {bm.max(grad_filter_sens):.6e}")
        print(f"- 平均值: {bm.mean(grad_filter_sens):.6e}")

        # 密度滤波
        filter_dens = Filter(FilterConfig(filter_type=1, filter_radius=2.4))
        filter_dens.initialize(mesh)
        
        objective_dens = ComplianceObjective(
            material_properties=material_properties,
            solver=solver,
            filter=filter_dens
        )
        
        grad_filter_dens = objective_dens.jac(design_vars=rho_elem[:], state_vars=displacement)
        print(f"\n密度滤波后的梯度信息:")
        print(f"- 形状 - {grad_filter_dens.shape}:\n, {grad_filter_dens}")
        print(f"- 最小值: {bm.min(grad_filter_dens):.6e}")
        print(f"- 最大值: {bm.max(grad_filter_dens):.6e}")
        print(f"- 平均值: {bm.mean(grad_filter_dens):.6e}")

        # Heaviside投影滤波
        filter_heav = Filter(FilterConfig(filter_type=2, filter_radius=2.0))
        filter_heav.initialize(mesh)
        
        # objective_heav = ComplianceObjective(
        #     material_properties=material_properties,
        #     solver=solver,
        #     filter=filter_heav
        # )
        
        # # 创建Heaviside滤波参数
        # beta = 1.0
        # rho_tilde = rho_elem[:]  # 这里简单使用原始密度作为示例
        
        # gradient_heav = objective_heav.jac(
        #     design_vars=rho_elem[:],
        #     filter_params={'beta': beta, 'rho_tilde': rho_tilde}
        # )
        # print(f"\nHeaviside投影滤波后的梯度信息:")
        # print(f"- 形状: {gradient_heav.shape}")
        # print(f"- 最小值: {bm.min(gradient_heav):.6e}")
        # print(f"- 最大值: {bm.max(gradient_heav):.6e}")
        # print(f"- 平均值: {bm.mean(gradient_heav):.6e}")

        # # 测试滤波矩阵属性
        # print("\n=== 测试滤波矩阵属性 ===")
        # H = filter_sens.H
        # Hs = filter_sens.Hs
        # print(f"滤波矩阵 H 的形状: {H.shape}")
        # print(f"滤波矩阵行和向量 Hs 的形状: {Hs.shape}")
        # print(f"Hs 的最小值: {bm.min(Hs):.6e}")
        # print(f"Hs 的最大值: {bm.max(Hs):.6e}")

    except Exception as e:
        print(f"梯度计算失败: {str(e)}")
        raise e

    

if __name__ == "__main__":
    test_compliance_objective()