import numpy as np
import mgis.behaviour as mgis_bv
from fealpy.csm.mfront import compile_mfront_file

file = compile_mfront_file('material/PostProcessingTest.mfront')

lib = "./libPostProcessingTest.so"  # 用实际路径替换

eps = 1.e-14
e = np.asarray([1.5e-2, 1.2e-2, 1.4e-2, 0., 0., 0.], dtype=np.float64) # 输入 应变
e2 = np.asarray([1.2e-2, 1.4e-2, 1.5e-2, 0., 0., 0.], dtype=np.float64) # 预期输出

h = mgis_bv.Hypothesis.Tridimensional # 表示是三维
b = mgis_bv.load(lib, 'PostProcessingTest', h) # 加载行为
postprocessings = b.getPostProcessingsNames()

m = mgis_bv.MaterialDataManager(b, 2) # 2 表示要处理的材料数量
mgis_bv.setMaterialProperty(m.s1, "YoungModulus", 150e9) # 设置材料属性
mgis_bv.setMaterialProperty(m.s1, "PoissonRatio", 0.3)
mgis_bv.setExternalStateVariable(m.s1, "Temperature", 293.15) # 设置外部状态变量
mgis_bv.update(m) # 更新材料数据
m.s1.gradients[0:] = e # 材料数据的梯度
m.s1.gradients[1:] = e
outputs = np.empty(shape=(2, 3), dtype=np.float64)
mgis_bv.executePostProcessing(outputs.reshape(6), m, "PrincipalStrain") # 后处理，获得主应变的值
print('principal strain:', outputs)
print('e2:', e2)
for i in range(0, 3):
    assert abs(outputs[0, i] - e2[i]) < eps, "invalid output value"
    assert abs(outputs[1, i] - e2[i]) < eps, "invalid output value"

K = mgis_bv.IntegrationType.IntegrationWithTangentOperator

