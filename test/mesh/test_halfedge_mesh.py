import numpy as np
import time
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh, QuadrangleMesh, PolygonMesh
from fealpy.mesh.halfedge_mesh import HalfEdgeMesh2d 
from fealpy.mesh import HalfEdgeMesh2d as HM

def animation_plot(method='poly'):
    if method in {'poly', 'quad'}:
        mesh = QuadrangleMesh.from_box([0, 2, 0, 1], 2, 1)
    if method in {'rg', 'nvb'}:
        mesh = TriangleMesh.from_box([0, 2, 0, 1], 2, 1)

    mesh = HalfEdgeMesh2d.from_mesh(mesh)
    #mesh = HalfEdgeMesh2d.from_box([0, 2, 0, 1], 0.3)
    mesh.init_level_info()
    NE = mesh.ds.NE

    r, h, N = 0.5, 1e-3, 10
    fig = plt.figure()
    axes = fig.gca()
    plt.ion()

    fff = 0
    for i in range(N):
        c = np.array([i*(2/N), 0.8])
        for k in range(10):
            node = mesh.entity('node')
            halfedge = mesh.entity('halfedge')
            pre = halfedge[:, 3]
            flag = np.linalg.norm(node-c, axis=1)<r
            isMarkedHEdge = flag[halfedge[:, 0]]&(~flag[halfedge[pre, 0]])
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
            isMarkedCell = isMarkedCell & (mesh.cell_area()>h**2)
            if (~isMarkedCell).all():
                break
            print('第', i, '轮, 加密', k, '次')
            if method == "poly":
                mesh.refine_poly(isMarkedCell)
            elif method == "rg":
                mesh.refine_triangle_rg(isMarkedCell)
            elif method == "nvb":
                mesh.refine_triangle_nvb(isMarkedCell)

            #if (i==1) & (k==2):
            #    fig = plt.figure()
            #    axes = fig.gca()
            #    mesh.add_plot(axes)
            #    #mesh.find_node(axes, showindex=True)
            #    mesh.find_cell(axes, showindex=True)
            #    mesh.add_halfedge_plot(axes, showindex=True)
            #    plt.show()

            plt.cla()
            mesh.add_plot(axes, linewidths = 0.4)
            #plt.savefig("fig_"+str(fff).zfill(3)+".png", dpi=400)
            fff+=1
            plt.pause(0.01)

        for k in range(10):
            halfedge = mesh.ds.halfedge
            pre = halfedge[:, 3]
            node = mesh.entity('node')
            flag = np.linalg.norm(node-c, axis=1)<r
            isMarkedHEdge = flag[halfedge[:, 0]]&(~flag[halfedge[pre, 0]])
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
            isMarkedCell = ~isMarkedCell & (mesh.celldata['level'][:]>0)
            if (~isMarkedCell).all():
                break
            print('第', i, '轮, 粗化', k, '次')
            if method == "poly":
                mesh.coarsen_poly(isMarkedCell)
            elif method == "rg":
                mesh.coarsen_triangle_rg(isMarkedCell)
            elif method == "nvb":
                mesh.coarsen_triangle_nvb(isMarkedCell)

            #if (i==0) & (k==1):
            #    fig = plt.figure()
            #    axes = fig.gca()
            #    mesh.add_plot(axes)
            #    #mesh.find_node(axes, showindex=True)
            #    mesh.find_cell(axes, showindex=True)
            #    mesh.add_halfedge_plot(axes, showindex=True)
            #    plt.show()

            plt.cla()
            mesh.add_plot(axes, linewidths = 0.4)
            #plt.savefig("fig_"+str(fff).zfill(3)+".png", dpi=400)
            fff+=1
            plt.pause(0.01)
    plt.ioff()
    plt.show()

def circle_plot(method='poly', plot=True):
    if method in {'poly', 'quad'}:
        mesh = QuadrangleMesh.from_box([0, 2, 0, 1], 10, 5)
    if method in {'rg', 'nvb'}:
        mesh = TriangleMesh.from_box([0, 2, 0, 1], 10, 5)
    mesh = HalfEdgeMesh2d.from_mesh(mesh)
    mesh = HalfEdgeMesh2d.from_box([0, 2, 0, 1], 0.3)
    mesh.init_level_info()
    NE = mesh.ds.NE

    r, h, N = 0.5, 1e-3, 10
    fig = plt.figure()
    axes = fig.gca()
    c = np.array([2*(2/N), 0.8])
    t0 = time.time()
    while True:
        s = time.time()
        halfedge = mesh.ds.halfedge
        pre = halfedge[:, 3]
        node = mesh.entity('node')
        flag = np.linalg.norm(node-c, axis=1)<r
        flag1 = flag[halfedge[:, 0]].astype(int)
        flag2 = flag[halfedge[pre, 0]].astype(int)
        isMarkedHEdge = flag1+flag2==1
        NC = mesh.number_of_cells()
        isMarkedCell = np.zeros(NC, dtype=np.bool_)
        isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
        isMarkedCell = isMarkedCell & (mesh.cell_area()>h**2)
        if (~isMarkedCell).all():
            break
        if method == "poly":
            mesh.refine_poly(isMarkedCell)
        elif method == "rg":
            mesh.refine_triangle_rg(isMarkedCell)
        elif method == "nvb":
            mesh.refine_triangle_nvb(isMarkedCell)
        e = time.time()

        mesh.add_plot(axes, linewidths = 0.2)
        #mesh.find_cell(axes, showindex=True)
        plt.pause(0.0001)
        print(e-s)
        #break
    t1 = time.time()
    print(t1-t0)

    plt.show()

def test_simple():
    mesh = QuadrangleMesh.from_box([0, 2, 0, 1], nx=2, ny=1)
    #mesh = TriangleMesh.from_one_triangle()
    hmesh = HalfEdgeMesh2d.from_mesh(mesh)
    hmesh.refine_poly(np.array([1, 0], dtype=np.bool_))
    hmesh.coarsen_poly(np.array([0, 0, 1, 1, 1], dtype=np.bool_))

    hmesh.refine_poly(np.array([1, 0], dtype=np.bool_))
    NC = hmesh.ds.number_of_cells()
    mark = np.ones(NC, dtype=np.bool_)
    mark[[4, 5, 6, 19]] = False
    hmesh.coarsen_poly(mark)

    hmesh.print()

    fig = plt.figure()
    axes = fig.gca()
    hmesh.add_plot(axes)
    hmesh.find_node(axes, showindex=True)
    hmesh.find_cell(axes, showindex=True)
    hmesh.add_halfedge_plot(axes, showindex=True)
    plt.show()

def rg_refine_test():
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=1, ny=1)
    hmesh = HalfEdgeMesh2d.from_mesh(mesh)

    isMarkedCell = np.array([0, 1], dtype=np.bool_)
    hmesh.refine_triangle_rg(isMarkedCell)

    isMarkedCell = np.array([0, 1, 1, 1, 0, 0], dtype=np.bool_)
    hmesh.refine_triangle_rg(isMarkedCell)

    hmesh.print()

    fig = plt.figure()
    axes = fig.gca()
    hmesh.add_plot(axes)
    hmesh.find_node(axes, showindex=True)
    hmesh.find_cell(axes, showindex=True)
    hmesh.add_halfedge_plot(axes, showindex=True)
    plt.show()

def rg_coarsen_test():
    mesh = TriangleMesh.from_box([0, 2, 0, 1], nx=2, ny=1)
    hmesh = HalfEdgeMesh2d.from_mesh(mesh)

    NC = hmesh.number_of_cells()
    isMarkedCell = np.zeros(NC, dtype=np.bool_)
    isMarkedCell[[0, 2]] = True
    hmesh.refine_triangle_rg(isMarkedCell)

    NC = hmesh.number_of_cells()
    isMarkedCell = np.zeros(NC, dtype=np.bool_)
    isMarkedCell[[0, 2, 5, 8, 4, 7, 9, 10]] = True
    hmesh.refine_triangle_rg(isMarkedCell)
    #hmesh.refine_triangle_rg()

    NC = hmesh.number_of_cells()
    isMarkedCell = np.zeros(NC, dtype=np.bool_)
    #isMarkedCell[:] = True
    #hmesh.coarsen_triangle_rg(isMarkedCell)

    hmesh.print()

    fig = plt.figure()
    axes = fig.gca()
    hmesh.add_plot(axes)
    #hmesh.find_node(axes, showindex=True)
    hmesh.find_cell(axes, showindex=True)
    #hmesh.add_halfedge_plot(axes, showindex=True)
    plt.show()

def nvb_refine_test():
    mesh = TriangleMesh.from_box([0, 2, 0, 1], nx=2, ny=1)
    hmesh = HalfEdgeMesh2d.from_mesh(mesh)

    isMarkedCell = np.array([0, 1, 0, 0], dtype=np.bool_)
    hmesh.refine_triangle_nvb(isMarkedCell)

    isMarkedCell = np.array([0, 1, 1, 1, 0, 0], dtype=np.bool_)
    hmesh.refine_triangle_nvb(isMarkedCell)

    hmesh.print()

    fig = plt.figure()
    axes = fig.gca()
    hmesh.add_plot(axes)
    hmesh.find_node(axes, showindex=True)
    hmesh.find_cell(axes, showindex=True)
    hmesh.add_halfedge_plot(axes, showindex=True)
    plt.show()

def nvb_coarsen_test():
    mesh = TriangleMesh.from_box([0, 2, 0, 1], nx=2, ny=1)
    hmesh = HalfEdgeMesh2d.from_mesh(mesh)

    NC = hmesh.number_of_cells()
    isMarkedCell = np.zeros(NC, dtype=np.bool_)
    isMarkedCell[[0, 2]] = True
    hmesh.refine_triangle_nvb(isMarkedCell)

    NC = hmesh.number_of_cells()
    isMarkedCell = np.zeros(NC, dtype=np.bool_)
    isMarkedCell[[0, 2, 4, 5]] = True
    hmesh.refine_triangle_nvb(isMarkedCell)

    NC = hmesh.number_of_cells()
    isMarkedCell = np.zeros(NC, dtype=np.bool_)
    isMarkedCell[[2, 4, 8, 10]] = True
    hmesh.coarsen_triangle_nvb(isMarkedCell)

    NC = hmesh.number_of_cells()
    isMarkedCell = np.zeros(NC, dtype=np.bool_)
    #isMarkedCell[:] = True
    #hmesh.coarsen_triangle_nvb(isMarkedCell)

    hmesh.print()

    fig = plt.figure()
    axes = fig.gca()
    hmesh.add_plot(axes)
    hmesh.find_node(axes, showindex=True)
    hmesh.find_cell(axes, showindex=True)
    hmesh.add_halfedge_plot(axes, showindex=True)
    plt.show()

def test_find_node_in_triangle_mesh():
    mesh = TriangleMesh.from_box([0, 2, 0, 1], nx=2, ny=1)
    hmesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)
    hmesh.refine_triangle_rg()
    hmesh.refine_triangle_rg()
    hmesh.refine_triangle_rg()
    hmesh.refine_triangle_rg()
    hmesh.refine_triangle_rg()
    hmesh.refine_triangle_rg()
    hmesh.refine_triangle_rg()
    hmesh.refine_triangle_rg()
    print(hmesh.number_of_cells())

    points = np.random.rand(160000, 2)
    #points = np.array([[1/3, 1/9.2]])
    points[:, 0] *= 2
    c, bc = hmesh.find_point_in_triangle_mesh(points)
    node = hmesh.entity('node')
    cell = np.array(hmesh.entity('cell'))
    points0 = np.einsum("cnd, cn->cd", node[cell[c]], bc)
    print(np.max(np.abs(points0-points)))

    fig = plt.figure()
    axes = fig.gca()
    hmesh.add_plot(axes)
    hmesh.find_cell(axes, showindex=True)
    axes.scatter(points[:, 0], points[:, 1], color='r')
    for i in range(len(points)):
        plt.annotate(i, points[i], textcoords="offset points", xytext=(0, 10),
                ha='center', color='r', fontsize=40)
    plt.show()

def test_cut_mesh():
    from fealpy.geometry import CircleCurve, Polygon
    #interface = CircleCurve([0, 0], np.sqrt(2))

    #points = np.array([[8/11, 0.3], [1.6, 1/3], [2, 4/11], [2.4, 1/3], [4-8/11, 0.5], 
    #                   [2.4, 8/11], [2, 2/3], [1.6, 2.5/3]])
    #points = np.array([[np.sqrt(2), 0.5], [np.sqrt(2)*1.2, 1/9],
    #    [np.sqrt(2)*2.1, 0.5], [np.sqrt(2)*1.5, 22/23]])
    #t = np.linspace(0, 2 * np.pi, 1000)
    #x = 16 * (np.sin(t) ** 3)
    #y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    #points = np.c_[x[:, None], y[:, None]]
    points = np.array([[311+2/3, 50], [290+0.625, 47+2/3], [225, 68+1/39], 
                 [206.25, 19+1/3], [200, 66+2/3], [178.125, 15+1/3], [175+1/3, 62+1/3],
                 [100+1/3, 69-1/4], [243.75, 82+1/3], [287.5, 52+1/3]])/100

    points = np.array([[311+2/3, 50], [290+0.625, 47+2/3], [225, 68+1/39], 
                 [206.25, 19+1/3], [198.05, 66+2/3], [178.125, 15+1/3], [175+1/3, 62+1/3],
                 [100+1/3, 69-1/4], [243.75, 82+1/3], [287.5, 52+1/3]])/100
    points[:, 1] = 1-points[:, 1]
    #fpidx = np.array([2, 6, 8, 9])
    #isMarked = np.ones(10, dtype=np.bool_)
    #isMarked[[2, 4, 6]] = False
    #points = points[isMarked]
    #points = np.array([[0.99, 0.50001], [1.233, 0.1001], [2.999, 0.8778123],
    #    [1.2, 0.602], [1.9, 0.502333]])

    interface = Polygon(points) 

    hmesh = HalfEdgeMesh2d.from_interface_cut_box(interface, [0, 4, 0, 1],
            2**12, 2**10, keep_feature=False)
    #hmesh = HalfEdgeMesh2d.from_mesh(QuadrangleMesh.from_box([0, 4, 0, 1],
    #    2**10, 2**8))

    v = interface(hmesh.node)
    hmesh.nodedata['val'] = np.sign(v)
    hmesh.to_vtk(fname='aaa.vtu')
    #fig = plt.figure()
    #axes = fig.gca()
    #hmesh.add_plot(axes)
    #axes.scatter(points[:, 0], points[:, 1], color='r')
    #for i in range(len(points)):
    #    plt.annotate(i, points[i], textcoords="offset points", xytext=(0, 10),
    #            ha='center', color='r', fontsize=40)
    #points = np.r_[points, points[0, None]]
    #axes.plot(points[:, 0], points[:, 1], color='r')
    #hmesh.find_node(axes, showindex=True)
    #plt.show()

def test_find_node():
    mesh = QuadrangleMesh.from_box([0, 2, 0, 1], nx=20, ny=10)
    hmesh = HalfEdgeMesh2d.from_mesh(mesh)

    points = np.random.rand(16, 2)
    #points = np.array([[1.2, 1/9]])
    points[:, 0] *= 2
    c = hmesh.find_point(points)
    print(c)

    fig = plt.figure()
    axes = fig.gca()
    hmesh.add_plot(axes)
    hmesh.find_cell(axes, showindex=True)
    #hmesh.add_halfedge_plot(axes, showindex=True)
    axes.scatter(points[:, 0], points[:, 1], color='r')
    for i in range(len(points)):
        plt.annotate(i, points[i], textcoords="offset points", xytext=(0, 10),
                ha='center', color='r', fontsize=40)
    plt.show()

def test_interpolation_data():
    def ff(p):
        return np.sin(p[..., 0])*np.sin(p[..., 1])
    mesh = TriangleMesh.from_box([0, 2, 0, 1], nx=2, ny=1)
    hmesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)
    hmesh.refine_triangle_rg()
    cell
    hmesh0 = copy.deepcopy(hmesh)

def test_interpolation_data(method='rg'):
    import copy
    import time
    mesh = TriangleMesh.from_box([0, 2, 0, 1], nx=20, ny=10)
    mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)
    #mesh = HalfEdgeMesh2d.from_box([0, 2, 0, 1], 0.3)

    def u(p): return np.sin(2*p[..., 1])*np.sin(2*p[..., 0])
    mesh.celldata['u'] = u(mesh.entity_barycenter('cell'))
    mesho = copy.deepcopy(mesh)

    r, h, N = 0.5, 1e-3, 10
    fff = 0
    for i in range(N):
        c = np.array([i*(2/N), 0.8])
        for k in range(10):
            fff+=1
            node = mesh.entity('node')
            halfedge = mesh.entity('halfedge')
            pre = halfedge[:, 3]
            flag = np.linalg.norm(node-c, axis=1)<r
            isMarkedHEdge = flag[halfedge[:, 0]]&(~flag[halfedge[pre, 0]])
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
            isMarkedCell = isMarkedCell & (mesh.cell_area()>h**2)
            if (~isMarkedCell).all():
                break

            if method == "rg":
                mesh.refine_triangle_rg(isMarkedCell)
            elif method == "nvb":
                mesh.refine_triangle_nvb(isMarkedCell)

            t0 = time.time()
            mesh.interpolation_cell_data(mesho, datakey=["u"])
            t1 = time.time()
            print("插值用时: ", t1-t0, "s")
            mesho = copy.deepcopy(mesh)

            mesh.to_vtk(fname='out_ref_'+str(fff).zfill(3)+".vtu")

        for k in range(10):
            fff+=1
            halfedge = mesh.ds.halfedge
            pre = halfedge[:, 3]
            node = mesh.entity('node')
            flag = np.linalg.norm(node-c, axis=1)<r
            isMarkedHEdge = flag[halfedge[:, 0]]&(~flag[halfedge[pre, 0]])
            NC = mesh.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool_)
            isMarkedCell[halfedge[isMarkedHEdge, 1]] = True
            isMarkedCell = ~isMarkedCell & (mesh.cell_area()<0.01)
            if (~isMarkedCell).all():
                break
            print('第', i, '轮粗化', k, '次')
            if method == "rg":
                mesh.coarsen_triangle_rg(isMarkedCell)
            elif method == "nvb":
                mesh.coarsen_triangle_nvb(isMarkedCell)

            t0 = time.time()
            mesh.interpolation_cell_data(mesho, datakey=["u"])
            t1 = time.time()
            print("插值用时: ", t1-t0, "s")
            mesho = copy.deepcopy(mesh)

            mesh.to_vtk(fname='out_ref_'+str(fff).zfill(3)+".vtu")


#animation_plot('nvb')
#circle_plot('nvb')
#rg_refine_test()
#rg_coarsen_test()
#test_find_node_in_triangle_mesh()
#nvb_refine_test()
#nvb_coarsen_test()
#test_cut_mesh()
#test_find_node()
#test_interpolation_data()
test_domain_with_hole()





