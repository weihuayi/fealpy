import numpy as np
from fealpy.mesh import HalfEdgeMesh2d, TriangleMesh
import matplotlib.pyplot as plt
from meshpy.triangle import MeshInfo, build
import copy

def f(node):
    node = node/np.linalg.norm(node, axis=-1).reshape((node.shape[:-1]+(1, )))
    return node

def dual_mesh_test():
    n = 20
    theta = np.linspace(0, np.pi*2, n, endpoint=False)
    node = np.c_[np.cos(theta), np.sin(theta)]
    line = np.c_[np.arange(n), (np.arange(n)+1)%n]
    mesh_int = MeshInfo()
    mesh_int.set_points(node)
    mesh_int.set_facets(line)


    mesh = build(mesh_int, max_volume = np.sqrt(3)*np.pi/640)
    node = np.array(mesh.points, dtype =np.float)
    cell = np.array(mesh.elements, dtype = np.int)

    #node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float_)
    #cell = np.array([[0, 1, 2], [0, 2, 3]])

    #node = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float_)
    #cell = np.array([[0, 1, 2]])

    mesh = TriangleMesh(node, cell)
    mesh = HalfEdgeMesh2d.from_mesh(mesh)
    #mesh.tri_uniform_refine(4)
    mesh0 = copy.deepcopy(mesh)
    mesh.to_dual_mesh(projection=f)

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    #mesh.add_halfedge_plot(axes, showindex=True)

    fig = plt.figure()
    axes = fig.gca()
    mesh0.add_plot(axes)
    #mesh.add_halfedge_plot(axes, showindex=True)
    plt.show()

def nvb_Refine_HB_test():
    node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float_)
    cell = np.array([[0, 1, 2], [0, 2, 3]])

    mesh = TriangleMesh(node, cell)
    mesh.uniform_refine()
    mesh = HalfEdgeMesh2d.from_mesh(mesh)

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_cell(axes, showindex=True)

    NC = mesh.number_of_all_cells()
    isMarkedCell = np.zeros(NC, dtype=np.bool_)
    isMarkedCell[[1, 5]] = True
    opt = {"HB": 1}
    mesh.refine_triangle_nvb(isMarkedCell, opt)
    print(opt['HB'])

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_cell(axes, showindex=True)
    plt.show()

def nvb_coarsen_HB_test():
    node = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float_)
    cell = np.array([[0, 1, 2], [0, 2, 3]])

    mesh = TriangleMesh(node, cell)
    mesh.uniform_refine()
    mesh = HalfEdgeMesh2d.from_mesh(mesh)

    NC = mesh.number_of_all_cells()
    isMarkedCell = np.zeros(NC, dtype=np.bool_)
    isMarkedCell[[1, 5, 8]] = True
    opt = {"HB": 1}
    mesh.refine_triangle_nvb(isMarkedCell, opt)

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_cell(axes, showindex=True)

    NC = mesh.number_of_all_cells()
    isMarkedCell = np.zeros(NC, dtype=np.bool_)
    isMarkedCell[[1, 10, 2, 9, 14, 8, 13]] = True
    mesh.coarsen_triangle_nvb(isMarkedCell, opt)
    print(opt['HB'])

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_cell(axes, showindex=True)
    plt.show()

nvb_coarsen_HB_test()
