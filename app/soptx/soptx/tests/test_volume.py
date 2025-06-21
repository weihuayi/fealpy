"""
测试 volume.py 中体积约束的功能
"""
from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh2d
from fealpy.functionspace import LagrangeFESpace

from soptx.opt import VolumeConstraint
from soptx.filter import Filter, FilterConfig

def test_volume_constraint():
    print("\n=== 体积约束测试 ===")
    
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
    
    # 2. 创建函数空间（用于设计变量）
    p = 1
    space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')
    
    # 3. 创建设计变量（初始为均匀分布）
    volfrac = 0.5  # 目标体积分数
    array = 0.4 * bm.ones(mesh.number_of_cells(), dtype=bm.float64)  # 初始密度小于目标体积分数
    rho_elem = space_D.function(array)

    print("\n=== 测试约束函数计算（不使用滤波器）===")
    try:
        # 创建体积约束对象（不使用滤波器）
        constraint = VolumeConstraint(
            mesh=mesh,
            volume_fraction=volfrac
        )

        # 计算约束值
        constraint_value = constraint.fun(design_vars=rho_elem[:])
        print(f"\n约束函数值（当前体积分数 - 目标体积分数）* 单元数:")
        print(f"- 约束值: {constraint_value:.6e}")

        # 计算约束梯度
        gradient = constraint.jac(design_vars=rho_elem[:])
        print(f"\n约束梯度信息:")
        print(f"- 形状 - {gradient.shape}:\n {gradient}")
        print(f"- 最小值: {bm.min(gradient):.6e}")
        print(f"- 最大值: {bm.max(gradient):.6e}")
        print(f"- 平均值: {bm.mean(gradient):.6e}")

        # 测试不同密度下的约束值
        print("\n=== 测试不同密度下的约束值 ===")
        # 测试较小密度
        rho_small = 0.1 * bm.ones_like(rho_elem[:])
        constraint_small = constraint.fun(design_vars=rho_small)
        print(f"密度=0.1时的约束值: {constraint_small:.6e}")
        
        # 测试较大密度
        rho_large = 0.9 * bm.ones_like(rho_elem[:])
        constraint_large = constraint.fun(design_vars=rho_large)
        print(f"密度=0.9时的约束值: {constraint_large:.6e}")

    except Exception as e:
        print(f"约束函数计算失败: {str(e)}")

    print("\n=== 测试不同类型的滤波器 ===")
    try:
        # 灵敏度滤波
        filter_sens = Filter(FilterConfig(filter_type=0, filter_radius=2.4))
        filter_sens.initialize(mesh)
        
        constraint_sens = VolumeConstraint(
            mesh=mesh,
            volume_fraction=volfrac,
            filter=filter_sens
        )
        
        grad_filter_sens = constraint_sens.jac(design_vars=rho_elem[:])
        print(f"\n灵敏度滤波后的梯度信息:")
        print(f"- 形状 - {grad_filter_sens.shape}:\n {grad_filter_sens}")
        print(f"- 最小值: {bm.min(grad_filter_sens):.6e}")
        print(f"- 最大值: {bm.max(grad_filter_sens):.6e}")
        print(f"- 平均值: {bm.mean(grad_filter_sens):.6e}")

        # 密度滤波
        filter_dens = Filter(FilterConfig(filter_type=1, filter_radius=2.4))
        filter_dens.initialize(mesh)
        
        constraint_dens = VolumeConstraint(
            mesh=mesh,
            volume_fraction=volfrac,
            filter=filter_dens
        )
        
        grad_filter_dens = constraint_dens.jac(design_vars=rho_elem[:])
        print(f"\n密度滤波后的梯度信息:")
        print(f"- 形状 - {grad_filter_dens.shape}:\n {grad_filter_dens}")
        print(f"- 最小值: {bm.min(grad_filter_dens):.6e}")
        print(f"- 最大值: {bm.max(grad_filter_dens):.6e}")
        print(f"- 平均值: {bm.mean(grad_filter_dens):.6e}")

        # # Heaviside投影滤波
        # filter_heav = Filter(FilterConfig(filter_type=2, filter_radius=2.4))
        # filter_heav.initialize(mesh)
        
        # constraint_heav = VolumeConstraint(
        #     mesh=mesh,
        #     volume_fraction=volfrac,
        #     filter=filter_heav
        # )
        
        # # 创建Heaviside滤波参数
        # beta = 1.0
        # rho_tilde = rho_elem[:]  # 这里简单使用原始密度作为示例
        
        # grad_filter_heav = constraint_heav.jac(
        #     design_vars=rho_elem[:],
        #     filter_params={'beta': beta, 'rho_tilde': rho_tilde}
        # )
        # print(f"\nHeaviside投影滤波后的梯度信息:")
        # print(f"- 形状: {grad_filter_heav.shape}")
        # print(f"- 最小值: {bm.min(grad_filter_heav):.6e}")
        # print(f"- 最大值: {bm.max(grad_filter_heav):.6e}")
        # print(f"- 平均值: {bm.mean(grad_filter_heav):.6e}")

    except Exception as e:
        print(f"滤波器测试失败: {str(e)}")
        raise e

if __name__ == "__main__":
    test_volume_constraint()