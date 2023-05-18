import numpy as np
import ipdb
from mgis import behaviour as mg


#ipdb.set_trace()
# 定义材料属性
young_modulus = 210.0e9
poisson_ratio = 0.3
material_properties = {'YoungModulus': young_modulus, 'PoissonRatio': poisson_ratio}

# 载入MFront模型
material_library = "./libsaint_venant_kirchhoff.so"  # 用实际路径替换
material_name = "SaintVenantKirchhoffElasticity"

# 载入MFront模型
#options = mg.FiniteStrainBehaviourOptions()
#options.hypothesis = mg.Hypothesis.TRIDIMENSIONAL
behaviour = mg.load(material_library, material_name, hypothesis=mg.Hypothesis.TRIDIMENSIONAL)

# 初始化状态变量和内部变量
state_variables = mg.create_behaviour_data(behaviour)

# 定义形变梯度张量（3x3）
deformation_gradient = np.array([
    [1.01, 0.0, 0.0],
    [0.0, 1.01, 0.0],
    [0.0, 0.0, 1.01]
])

# 更新材料状态
mg.integrate(behaviour, state_variables, material_properties, deformation_gradient, mg.IntegrationType.FiniteStrain)

# 提取更新后的应力张量
stress = state_variables.s1.gradients["Stress"]

print("Stress tensor: ", stress)
