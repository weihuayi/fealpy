import numpy as np
from mgis import behaviour as mgis_bv
from fealpy.csm.mfront import compile_mfront_file

file = compile_mfront_file('material/PhaseFieldDisplacementSpectralSplit.mfront')

lib = "./libPhaseFieldDisplacementSpectralSplit.so"  # 用实际路径替换

# 定义应变张量
eto = np.zeros(6)
eto[1] = 1.0
eto[2] = 1.0
h = mgis_bv.Hypothesis.Tridimensional # 表示是三维

# 加载行为模型
b = mgis_bv.load(lib, "PhaseFieldDisplacementSpectralSplit", h)

# 设置材料属性
m = mgis_bv.MaterialDataManager(b, 1) # 2 表示积分点的个数
E = 150e9
nu = 0.3

lam = nu*E/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))

print('lam:', lam)
print('mu:', mu)
print('2mu+lam', 2*mu+lam)

# 设置材料属性
m = mgis_bv.MaterialDataManager(b, 2) # 2 表示要处理的材料数量
mgis_bv.setMaterialProperty(m.s1, "YoungModulus", 150e9) # 设置材料属性
mgis_bv.setMaterialProperty(m.s1, "PoissonRatio", 0.3)
mgis_bv.setExternalStateVariable(m.s1, "Temperature", 293.15) #设置外部状态变量
mgis_bv.setExternalStateVariable(m.s1, "Damage", 0.1) #设置外部状态变量
# 初始化局部变量
mgis_bv.update(m) # 更新材料数据
m.s0.internal_state_variables[0] = 1e15
m.s1.gradients[0:] = eto
it = mgis_bv.IntegrationType.IntegrationWithTangentOperator
#it = mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator
dt = 0.1 
mgis_bv.integrate(m, it, dt, 0, m.n)

#idx = mgis_bv.getVariableSize(b.thermodynamic_forces[0], h)
sig = m.s1.thermodynamic_forces
H = m.s1.internal_state_variables[0]

print("H:", H)
print("Predicted Stress:", idx, m.s1.thermodynamic_forces)
print("Tangent Stiffness:", m.K)


