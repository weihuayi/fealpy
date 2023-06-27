import numpy as np
import mgis.behaviour as mgis_bv
from fealpy.csm.mfront import compile_mfront_file

file = compile_mfront_file('material/PostProcessingTest.mfront')

lib = "./libPostProcessingTest.so"  # 用实际路径替换

eps = 1.e-14
e = np.asarray([1.3e-2, 1.2e-2, 1.4e-2, 0., 0., 0.],
                  dtype=numpy.float64) # 输入 应变
e2 = np.asarray([1.2e-2, 1.3e-2, 1.4e-2, 0., 0., 0.],
                   dtype=numpy.float64) # 预期输出

h = mgis_bv.Hypothesis.Tridimensional # 表示是三维
b = mgis_bv.load(lib, 'PostProcessingTest', h) # 加载行为
