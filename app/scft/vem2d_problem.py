import numpy as np
from SCFTVEMModel2d import SCFTVEMModel2d, scftmodel2d_options
from fealpy.functionspace.vem_space import VirtualElementSpace2d
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh
from fealpy.mesh import Quadtree, QuadrangleMesh, HalfEdgeMesh2d

__doc__ = """
该文件包含了所有用来测试的问题模型
"""


def init_mesh(n=4, h=4):
    """
    生成初始的网格
    """
    node = np.array([
        (0, 0), (h, 0), (h, h), (0, h)], dtype=np.float64)

    cell = np.array([(0, 1, 2, 3)], dtype=np.int_)
    mesh = Quadtree(node, cell)
    mesh.uniform_refine(n)
    return mesh

def halfedgemesh(n=4, h=4):
    """
    半边数据结构的网格
    """
    node = np.array([
        (0, 0), (h, 0), (h, h), (0, h)], dtype=np.float64)
    cell = np.array([(0, 1, 2, 3)], dtype=np.int_)

    mesh = QuadrangleMesh(node, cell)
    mesh = HalfEdgeMesh2d.from_mesh(mesh)
    mesh.uniform_refine(n)
    return mesh

def quadmesh(n=10, L=12):
    box = [0, L, 0, L]
    qmesh = StructureQuadMesh(box, n, n)
    node = qmesh.node
    cell = qmesh.ds.cell
    quadtree = Quadtree(node, cell)
    return quadtree

def quad_model(fieldstype=3, n=10, L=12, options=None):
    quadtree = quadmesh(n)
    NN = quadtree.number_of_nodes()
    print('The number of mesh:', NN)
    mesh = quadtree.to_pmesh()
    obj = SCFTVEMModel(mesh, options=options)
    mu = obj.init_value(fieldstype=fieldstype)  # get initial value
    problem = {'objective': obj, 'x0': mu, 'quadtree': quadtree}
    return problem

def plane_model(fieldstype=3,n=5,options=None):
    quadtree = init_mesh(n, h=12)
    NN = quadtree.number_of_nodes()
    print('The number of mesh:', NN)
    mesh = quadtree.to_pmesh()
    obj = SCFTVEMModel(mesh, options=options)
    mu = obj.init_value(fieldstype=fieldstype)  # get initial value
    problem = {'objective': obj, 'x0': mu, 'quadtree': quadtree}
    return problem


def adaptive_model(fieldstype, options):

    quadtree = init_mesh(n=6, h=20)
    mesh = quadtree.to_pmesh()
    obj = SCFTVEMModel(mesh, options=options)
    mu = obj.init_value(fieldstype=fieldstype)  # get initial value
    problem = {'objective': obj, 'x0': mu, 'quadtree': quadtree}
    return problem

def converge_model(fieldstype, options,dataname):
    import scipy.io as scio
    data = scio.loadmat(dataname)
    rho = data['rho']
    mu = data['mu']
    q = rho.copy()
    q[:,0] = data['q0'][:,-1]
    q[:,1] = data['q1'][:,-1]

    quadtree = init_mesh(n=7, h=12)
    mesh = quadtree.to_pmesh()
    obj = SCFTVEMModel(mesh, options=options)
    problem = {'objective': obj, 'x0': mu, 'quadtree': quadtree, 'rho': rho,
            'q': q}
    return problem

def converge_apt_model(fieldstype, options, dataname):
    import scipy.io as scio
    data = scio.loadmat(dataname)
    rho = data['rho']
    mu = data['mu']
    import pickle
    quadtree1 = open('7558quadtree.bin','rb')
    quadtree = pickle.load(quadtree1)
    mesh = quadtree.to_pmesh()
    obj = SCFTVEMModel(mesh, options=options)
    problem = {'objective': obj, 'x0': mu, 'quadtree': quadtree, 'rho': rho}
    return problem
