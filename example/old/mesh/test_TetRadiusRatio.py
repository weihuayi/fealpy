import numpy as np
from fealpy.mesh import MeshFactory, TetrahedronMesh
from TetRadiusRatio import TetRadiusRatio
from scipy.sparse import bmat
import matplotlib.pyplot as plt

def test0():
    '''
    一个四面体单元优化
    '''
    node = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5*np.sqrt(3)/2, 0.0],
        [0.5, np.sqrt(3)/6, 0.5*np.sqrt(2/3)]], dtype=np.float64)
    cell = np.array([[0, 1, 2, 3]],dtype=np.int_)
    mesh = TetrahedronMesh(node, cell)
    opt = TetRadiusRatio(mesh)
    isFreeNode = np.array([0,0,1,1],dtype=np.bool_)
    node = mesh.entity('node')
    for i in range(30):
        A,B0,B1,B2 = opt.grad()
        #opt.BlockJacobi(node,A,B0,B1,B2,isFreeNode)
        opt.Jacobi(node,A,B0,B1,B2,isFreeNode)
        print("q:",opt.get_quality())
    return mesh
def test_two_tet():
    '''
    两个四面体单元优化
    '''
    node = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5*np.sqrt(3)/2, 0.0],
        [0.5, np.sqrt(3)/6, 0.5*np.sqrt(2/3)],
        [0.5, np.sqrt(3)/6,-np.sqrt(2/3)]], dtype=np.float64)
    cell = np.array([[0, 1, 2, 3],[0,1,2,4]],dtype=np.int_)
    mesh = TetrahedronMesh(node, cell)
    opt = TetRadiusRatio(mesh)
    isFreeNode = np.array([0,0,1,1,1],dtype=np.bool_)
    node = mesh.entity('node')
    for i in range(20):
        A,B0,B1,B2 = opt.grad()
        opt.BlockJacobi(node,A,B0,B1,B2,isFreeNode)
        print("q:",opt.get_quality())
    return mesh

def test1(method='jacobi'):
    mesh = TetrahedronMesh.from_one_tetrahedron()
    mesh.uniform_refine(n=3)
    opt = TetRadiusRatio(mesh)
    q = opt.get_quality()
    print('初始网格：minq=',np.min(q),'avgq=',np.mean(q),'maxq=',np.max(q))
    opt.iterate_solver(method=method)
    q = opt.get_quality()
    print('优化网格：minq=',np.min(q),'avgq=',np.mean(q),'maxq=',np.max(q))
    return mesh

def test2(method='jacobi'):
    mesh = TetrahedronMesh.from_one_tetrahedron()
    mesh.uniform_refine(n=3)
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    node[cell[-1,2]]=np.average(node[cell[-1]],axis=0)
    mesh.uniform_refine(n=2)
    opt = TetRadiusRatio(mesh)
    q = opt.get_quality()
    print('初始网格：minq=',np.min(q),'avgq=',np.mean(q),'maxq=',np.max(q))
    opt.iterate_solver(method=method)
    q = opt.get_quality()
    print('优化网格：minq=',np.min(q),'avgq=',np.mean(q),'maxq=',np.max(q))
    return mesh

def test3(method='jacobi'):
    mesh = TetrahedronMesh.from_one_tetrahedron()
    mesh.uniform_refine(n=2)
    isBdNode = mesh.ds.boundary_node_flag()
    node = mesh.entity('node')
    np.random.seed(0)
    node[~isBdNode] += 0.05*np.random.rand(node[~isBdNode].shape[0],
            node[~isBdNode].shape[1])

    mesh.uniform_refine(n=4)
    opt = TetRadiusRatio(mesh)
    q = opt.get_quality()
    print('初始网格：minq=',np.min(q),'avgq=',np.mean(q),'maxq=',np.max(q))
    opt.iterate_solver(method=method)
    
    return mesh

def test5():
    mesh = TetrahedronMesh.from_one_tetrahedron()
    mesh.uniform_refine(n=3)
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    node[cell[-1,2]]=np.average(node[cell[-1]],axis=0)
    mesh.uniform_refine(n=2)
    opt = TetRadiusRatio(mesh)
    q = opt.get_quality()
    print('初始网格：minq=',np.min(q),'avgq=',np.mean(q),'maxq=',np.max(q))
    opt.iterate_solver(method='Bjacobi')
    return mesh


def test_gmsh():
    mesh = TetrahedronMesh.from_unit_sphere_gmsh(h=0.1)
    opt = TetRadiusRatio(mesh)
    q = opt.get_quality()
    print('初始网格：minq=',np.min(q),'avgq=',np.mean(q),'maxq=',np.max(q))
    opt.iterate_solver(method='Bjacobi')
    return mesh




#mesh = test0()
#mesh = test1()
#mesh = test2(method='Bjacobi')
#mesh = test3(method='Bjacobi')
mesh = test_gmsh()
#mesh = test_two_tet()
#mesh = test5()
'''
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
#mesh.add_plot(axes, threshold=lambda p: p[..., 0] > 0.0)
mesh.add_plot(axes)
plt.show()
'''
