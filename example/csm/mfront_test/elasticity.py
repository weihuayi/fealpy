import numpy as np
import mgis.behaviour as mgis_bv
from fealpy.csm.mfront import compile_mfront_file

file = compile_mfront_file('material/Elasticity.mfront')

lib = "./libElasticity.so"  # 用实际路径替换

# 定义应变张量
eto = np.zeros(4)
eto[0] = 3.0
eto[1] = 2.0
#eto[2] = 2.0
eto[3] = 1.0
h = mgis_bv.Hypothesis.PlaneStrain # 平面
#h = mgis_bv.Hypothesis.Tridimensional # 表示是三维

# 加载行为模型
b = mgis_bv.load(lib, "Elasticity", h)

# 设置材料属性
E = 150e9
nu = 0.3

lam = nu*E/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))

print('lam:', lam)
print('mu:', mu)
print('2mu+lam', 2*mu+lam)
print('0:', eto[0]*2*mu+(eto[0]+eto[1])*lam)
print('0:', eto[1]*2*mu+(eto[0]+eto[1])*lam)
print('1:', eto[3]*2*mu)
print('2:', eto[2]*2*mu)
print('3:', eto[3]*2*mu+(eto[0]+eto[3])*lam)

m = mgis_bv.MaterialDataManager(b, 1) # 2 表示要处理的材料数量
mgis_bv.setMaterialProperty(m.s1, "YoungModulus", 150e9) # 设置材料属性
mgis_bv.setMaterialProperty(m.s1, "PoissonRatio", 0.3)
mgis_bv.setExternalStateVariable(m.s1, "Temperature", 293.15) # 设置外部状态变量



# 初始化局部变量
mgis_bv.update(m) # 更新材料数据
#m.s0.gradients[0:] = eto
m.s1.gradients[0:] = eto
it = mgis_bv.IntegrationType.IntegrationWithTangentOperator
#it = mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator
dt = 0.1 
mgis_bv.integrate(m, it, dt, 0, m.n)

idx = mgis_bv.getVariableSize(b.thermodynamic_forces[0], h)
print(b.thermodynamic_forces[0], idx)
# 输出结果
print('m.n', m.n)
print("Predicted Stress:", m.s1.thermodynamic_forces)
print("Tangent Stiffness:", m.K)
print("predicted stress:", m.K@eto)
