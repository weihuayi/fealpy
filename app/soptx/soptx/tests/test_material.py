from fealpy.backend import backend_manager as bm

from fealpy.mesh import UniformMesh2d

from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import TensorFunctionSpace


from soptx.material import (
    ElasticMaterialConfig,
    ElasticMaterialProperties,
    SIMPInterpolation
)

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

material_config = ElasticMaterialConfig(
                                        elastic_modulus=1.0,
                                        minimal_modulus=1e-9,
                                        poisson_ratio=0.3,
                                        plane_assumption="plane_stress"
                                    )
print("\n=== 材料配置测试 ===")
print(f"弹性模量: {material_config.elastic_modulus}")
print(f"最小模量: {material_config.minimal_modulus}")
print(f"泊松比: {material_config.poisson_ratio}")
print(f"平面假设: {material_config.plane_assumption}")

interpolation_model = SIMPInterpolation(penalty_factor=3.0)
print("\n=== 插值模型测试 ===")
print(f"插值模型名称: {interpolation_model.name}")
print(f"惩罚因子: {interpolation_model.penalty_factor}")


volfrac = 0.5
array = volfrac * bm.ones(nx * ny, dtype=bm.float64)
rho_elem = space_D.function(array)
material_properties = ElasticMaterialProperties(
                                                config=material_config,
                                                rho=rho_elem[:],
                                                interpolation_model=interpolation_model
                                            )
print("\n=== 材料属性测试 ===")
# 测试配置获取
config = material_properties.config
print(f"获取的配置 - 弹性模量: {config.elastic_modulus}")
print(f"获取的配置 - 最小模量: {config.minimal_modulus}")
print(f"获取的配置 - 泊松比: {config.poisson_ratio}")
print(f"获取的配置 - 平面假设: {config.plane_assumption}")


# 测试密度场获取
rho_get = material_properties.rho
print(f"获取的密度场均值: {bm.mean(rho_get)}")

# 测试插值模型获取
interp_model = material_properties.interpolation_model
print(f"获取的插值模型名称: {interp_model.name}")

# 测试基础材料属性获取
base_material = material_properties.base_elastic_material

print(f"基础材料弹性模量: {base_material.elastic_modulus}")
print(f"基础材料泊松比: {base_material.poisson_ratio}")


# 6. 测试材料模型计算
print("\n=== 材料模型计算测试 ===")
# 计算插值后的杨氏模量
E = material_properties.material_model()
print(f"插值后的杨氏模量均值: {bm.mean(E)}")
print(f"插值后的杨氏模量最大值: {bm.max(E)}")
print(f"插值后的杨氏模量最小值: {bm.min(E)}")

# 计算材料属性导数
dE = material_properties.material_model_derivative()
print(f"材料属性导数均值: {bm.mean(dE)}")

# 7. 测试弹性矩阵计算
print("\n=== 弹性矩阵计算测试 ===")
D = material_properties.elastic_matrix()
print(f"弹性矩阵形状: {D.shape}")
print(f"弹性矩阵的一个示例元素: \n{D[0]}")

print("-----------------------")