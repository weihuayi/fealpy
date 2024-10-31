import numpy as np
from mgis import behaviour as mg
from fealpy.csm.mfront import compile_mfront_file

import ipdb


file = compile_mfront_file('material/saint_venant_kirchhoff.mfront')

# 加载共享库
behaviour_library = "./libsaint_venant_kirchhoff.so"  # 用实际路径替换
behaviour_name = 'SaintVenantKirchhoffElasticity'

# 创建有限应变行为选项
options = mg.FiniteStrainBehaviourOptions()
options.stress_measure = mg.FiniteStrainBehaviourOptionsStressMeasure.CAUCHY
options.tangent_operator = mg.FiniteStrainBehaviourOptionsTangentOperator.DS_DEGL

# 加载行为
h = mg.Hypothesis.Tridimensional
b = mg.load(options, behaviour_library, behaviour_name, h)

#ipdb.set_trace()
print('source:', b.source)
print('behaviour:', b.behaviour)
print('hypothesis:', b.hypothesis)
print(b.tfel_version)
print(b.esvs)
print(mg.getArraySize(b.esvs, b.hypothesis))
print(b.esvs[0].name)
print(b.esvs[0].type)

d = mg.BehaviourData(b)
print('s0:', d.s0.external_state_variables)
print('s1:', d.s1)

# 设置材料属性
young = 200000.0
nu = 0.3

n = 1
mstate = mg.MaterialDataManager(b, n)

# 为行为设置 Young's modulus 和 Poisson's ratio
mstate.s0.setMaterialProperty('YoungModulus', young)
mstate.s0.setMaterialProperty('PoissonRatio', nu)

# 执行计算
strain = np.zeros((6,), dtype=np.float64) # 设置应变张量（此处示例使用零张量）
mstate.s0.e0 = strain #设置应变

# 计算切线算子
#tangent_operator = mg.Tensor()  # 创建一个Tensor对象以存储切线算子
#mstate.s1.k = tangent_operator  # 设置切线算子

# 执行计算
kk = mstate.K
mg.compute_tangent_operator(mstate)

# 获取切线算子信息
print('切线算子:')
print('Size:', tangent_operator.size())
print('Values:', tangent_operator.values())

# 获取状态变量信息
state_variables = mstate.s1.iv1
print('状态变量:')
for i, sv in enumerate(state_variables):
    print('State Variable', i)
    print('Name:', sv.name)
    print('Type:', sv.type)
    print('Size:', mg.getArraySize(sv, b.hypothesis))
    print('Values:', sv.values())


