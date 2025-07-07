import argparse
from fealpy.backend import backend_manager as bm
from fealpy.csm.fem import ElastoplasticityFEMModel

# 参数解析
parser = argparse.ArgumentParser(description="""
        用有限元方法计算弹塑性问题的位移
        """)

# 解析参数
options = vars(parser.parse_args())

bm.set_backend('numpy')