import numpy as np
from mgis import behaviour as mg
import ipdb

ipdb.set_trace()


# 加载共享库
behaviour_library = './libBehaviour.so'  # 或者是你的库文件的实际路径
behaviour_name = 'SaintVenantKirchhoffElasticity'

# 创建有限应变行为选项
options = mg.FiniteStrainBehaviourOptions()
options.stress_measure = mg.FiniteStrainBehaviourOptionsStressMeasure.CAUCHY
options.tangent_operator = mg.FiniteStrainBehaviourOptionsTangentOperator.DS_DEGL

# 加载行为
behaviour = mg.load(options, behaviour_library, behaviour_name, mg.Hypothesis.Tridimensional)

# 设置材料属性
young = 200000.0
nu = 0.3

n = 1
mstate = mg.MaterialDataManager(behaviour, n)

# 为行为设置 Young's modulus 和 Poisson's ratio
mstate.s0.setMaterialProperty('YoungModulus', young)
mstate.s0.setMaterialProperty('PoissonRatio', nu)


