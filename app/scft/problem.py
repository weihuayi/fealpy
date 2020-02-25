import numpy as np
from PRISMSCFTFEMModel import PRISMSCFTFEMModel, pscftmodel_options
from fealpy.mesh import PrismMesh

__doc__ = """
该文件包含了所有用来测试的问题模型
"""


def init_mesh(n=4, h=10):
    """
    生成初始的网格
    """
    node = np.array([
             [0, 0, 0],
             [h, 0, 0],
             [0, h, 0],
             [0, 0, h],
             [h, 0, h],
             [0, h, h]], dtype=np.float)
    cell = np.array([[0, 1, 2, 3, 4,  5]], dtype=np.int)
    pmesh = PrismMesh(node, cell)
    pmesh.uniform_refine(n)
    return pmesh

def prism_model(fieldstype=1, n=5, options=None):
    mesh = init_mesh(n, h=12)
    NN = mesh.number_of_nodes()
    print('The number of mesh:', NN)
    obj = PRISMSCFTFEMModel(mesh, options=options)
    mu = obj.init_value(fieldstype=fieldstype)  # get initial value
    problem = {'objective': obj, 'x0': mu, 'mesh': mesh}
    return problem
