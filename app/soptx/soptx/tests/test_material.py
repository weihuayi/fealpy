'''
Author: heliang
Date: 2025-2-06
'''
from fealpy.backend import backend_manager as bm

from fealpy.mesh import UniformMesh2d

from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace


from soptx.material import (ElasticMaterialConfig,
                            ElasticMaterialProperties)

nx, ny = 10, 10
extent = [0, nx, 0, ny]
h = [1.0, 1.0]
origin = [0.0, 0.0]
mesh = UniformMesh2d(extent=extent, h=h, origin=origin, 
                    ipoints_ordering='yx', flip_direction='y', 
                    device='cpu')
NC = mesh.number_of_cells()
    
p = 1
space_C = LagrangeFESpace(mesh=mesh, p=p, ctype='C')
tensor_space_C = TensorFunctionSpace(space_C, (-1, 2))
space_D = LagrangeFESpace(mesh=mesh, p=p-1, ctype='D')

# 1. 配置材料属性
material_config = ElasticMaterialConfig(
                        elastic_modulus=1,             # 杨氏模量 (Pa)
                        minimal_modulus=1e-9,          # 最小杨氏模量 (Pa)
                        poisson_ratio=0.3,             # 泊松比
                        plane_assumption="3D",         # 3D 假设
                        interpolation_model="SIMP",    # 选择插值模型为 SIMP
                        penalty_factor=3.0             # 惩罚因子
                    )
print("\n=== 材料配置测试 ===")
print(f"弹性模量: {material_config.elastic_modulus}")
print(f"最小模量: {material_config.minimal_modulus}")
print(f"泊松比: {material_config.poisson_ratio}")
print(f"平面假设: {material_config.plane_assumption}")
print(f"插值模型: {material_config.interpolation_model}")
print(f"惩罚因子: {material_config.penalty_factor}")

# 2. 创建材料属性实例
material_properties = ElasticMaterialProperties(config=material_config)

# 3. 假设我们有一个材料的密度场 (可以是任何适合的 TensorLike 对象)
# 这里假设密度场为均匀的 1，表示完全的实心材料。
volfrac = 1
array = volfrac * bm.ones(nx * ny, dtype=bm.float64)
rho_elem = space_D.function(array)

# 4. 计算杨氏模量
elastic_modulus = material_properties.calculate_elastic_modulus(density=rho_elem[:])
print(f"Elastic Modulus:\n {elastic_modulus}")

# 5. 计算杨氏模量对密度的导数（灵敏度）
elastic_modulus_derivative = material_properties.calculate_elastic_modulus_derivative(density=rho_elem[:])
print(f"Elastic Modulus Derivative:\n, {elastic_modulus_derivative}")

# 6. 获取基础材料实例（E=1）
base_material = material_properties.get_base_material()
print("Base Material:", base_material)