import numpy as np
from mgis import behaviour as mg
import ipdb
from mfront_duild import mfront_build

#ipdb.set_trace()

# 调用编译函数并传入.mfront文件名
so_fname = mfront_build.compile_mfront_file("material/saint_venant_kirchhoff.mfront")

# 加载共享库
behaviour_library = 'so_fname'  # 或者是你的库文件的实际路径
behaviour_name = 'SaintVenantKirchhoffElasticity'

# 创建有限应变行为选项
options = mg.FiniteStrainBehaviourOptions()
options.stress_measure = mg.FiniteStrainBehaviourOptionsStressMeasure.CAUCHY
options.tangent_operator = mg.FiniteStrainBehaviourOptionsTangentOperator.DS_DEGL

# 加载行为
h = mg.Hypothesis.Tridimensional
b = mg.load(options, behaviour_library, behaviour_name, h)

print(b.source)
print(b.behaviour)
print(b.hypothesis)
print(b.tfel_version)
print(b.esvs)
print(mg.getArraySize(b.esvs, b.hypothesis))
print(b.esvs[0].name)
print(b.esvs[0].type)

d = mg.BehaviourData(b)
print(d.s0.external_state_variables)
print(d.s1)

# 设置材料属性
young = 200000.0
nu = 0.3

n = 1
mstate = mg.MaterialDataManager(b, n)

# 为行为设置 Young's modulus 和 Poisson's ratio
mstate.s0.setMaterialProperty('YoungModulus', young)
mstate.s0.setMaterialProperty('PoissonRatio', nu)


