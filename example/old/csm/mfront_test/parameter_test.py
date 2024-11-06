import mgis.behaviour as mgis_bv
from fealpy.csm.mfront import compile_mfront_file

yg = 150e9
nu = 0.3
eps = 1.e-14
# path to the test library

file = compile_mfront_file('material/ParameterTest.mfront')

# 加载共享库
lib = "./libParameterTest.so"  # 用实际路径替换
behaviour_name = 'ParameterTest'

h = mgis_bv.Hypothesis.Tridimensional
# 加载行为
b = mgis_bv.load(lib, 'ParameterTest', h)
print('h:', h)
print('params:', b.params)
print('source:', b.source)
# 获取参数的默认值
yg_v = mgis_bv.getParameterDefaultValue(b, "YoungModulus")
nu_v = mgis_bv.getParameterDefaultValue(b, "PoissonRatio")

print('yg:', yg, yg_v)
print('nu:', nu, nu_v)
