import math
import numpy as np
import mgis.behaviour as mgis_bv
import mgis.model
from fealpy.csm.mfront import compile_mfront_file

import ipdb

ipdb.set_trace()

file = compile_mfront_file('material/ode_rk54.mfront')

# 测试库的路径
lib = "./libode_rk54.so"  # 用实际路径替换

# 建模假设
h = mgis_bv.Hypothesis.Tridimensional
# 加载行为
model = mgis.model.load(lib, 'ode_rk54', h)
# 参数 A 的默认值
A = model.getParameterDefaultValue('A')
# 积分点数量
nig = 100
# 材料数据管理器
m = mgis_bv.MaterialDataManager(model, nig)
# x 在状态变量数组中的索引
o = mgis_bv.getVariableOffset(model.isvs, 'x', h)
# 时间步长增量
dt = 0.1
# 设置温度
T = 293.15 * np.ones(nig)
# 存储类型
Ts = mgis_bv.MaterialStateManagerStorageMode.ExternalStorage
mgis_bv.setExternalStateVariable(m.s1, 'Temperature', T, Ts)
# x 的初始值
for n in range(0, nig):
    m.s1.internal_state_variables[n][o] = 1
# 将 s1 复制到 s0
mgis_bv.update(m)
# 第一个积分点的索引
ni = 0
# 最后一个积分点的索引
ne = nig - 1
# 第一个积分点的 x 值
xi = [m.s0.internal_state_variables[ni][o]]
# 最后一个积分点的 x 值
xe = [m.s0.internal_state_variables[ne][o]]
# 积分
for i in range(0, 10):
    it = mgis_bv.IntegrationType.IntegrationWithoutTangentOperator
    print('it:', it)
    mgis_bv.integrate(m, it, dt, 0, m.n)
    print('m.n:', m.n)
    mgis_bv.update(m)
    xi.append(m.s1.internal_state_variables[ni][o])
    xe.append(m.s1.internal_state_variables[ne][o])

# 检查
eps = 1.e-10
t = 0
for i in range(0, 11):
    x_ref = math.exp(-A * t)
    assert abs(xi[i] - x_ref) < eps
    assert abs(xe[i] - x_ref) < eps
    t = t + dt

