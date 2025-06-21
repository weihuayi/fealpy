import numpy as np
import pytest
import matplotlib.pyplot as plt

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh import IntervalMesh
from fealpy.mesh import TriangleMesh
from fealpy.mesh import TetrahedronMesh
from fealpy.mesh.halfedge_mesh import HalfEdgeMesh2d 

@pytest.mark.parametrize("p", range(1, 10))
def test_interval_mesh(p):
    mesh = IntervalMesh.from_one_triangle()
    space = LagrangeFESpace(mesh, p=p)

    assert space.geo_dimension() == 2
    assert space.top_dimension() == 2
    assert space.number_of_global_dofs() ==  (p+1)*(p+2)/2
    assert space.number_of_local_dofs() == (p+1)*(p+2)/2

    mesh = TriangleMesh.from_unit_square(nx=2, ny=2)
    space = LagrangeFESpace(mesh, p=p)
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    assert space.geo_dimension() == 2
    assert space.top_dimension() == 2
    assert space.number_of_global_dofs() == NN + (p-1)*NE + (p-2)*(p-1)/2*NC  
    assert space.number_of_local_dofs() == (p+1)*(p+2)/2

@pytest.mark.parametrize("p", range(1, 10))
def test_triangle_mesh(p):
    mesh = TriangleMesh.from_one_triangle()
    space = LagrangeFESpace(mesh, p=p)

    assert space.geo_dimension() == 2
    assert space.top_dimension() == 2
    assert space.number_of_global_dofs() ==  (p+1)*(p+2)/2
    assert space.number_of_local_dofs() == (p+1)*(p+2)/2

    mesh = TriangleMesh.from_unit_square(nx=2, ny=2)
    space = LagrangeFESpace(mesh, p=p)
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    assert space.geo_dimension() == 2
    assert space.top_dimension() == 2
    assert space.number_of_global_dofs() == NN + (p-1)*NE + (p-2)*(p-1)/2*NC  
    assert space.number_of_local_dofs() == (p+1)*(p+2)/2

@pytest.mark.parametrize("p", range(10))
def test_tetrahedron_mesh(p):
    mesh = TetrahedronMesh.from_one_tetrahedron(meshtype='equ')
    space = LagrangeFESpace(mesh, p=p)

    assert space.geo_dimension() == 3
    assert space.top_dimension() == 3
    assert space.number_of_global_dofs() ==  (p+3)*(p+2)*(p+1)//6
    assert space.number_of_local_dofs() == (p+3)*(p+2)*(p+1)//6

    mesh = TetrahedronMesh.from_unit_cube(nx=10, ny=10, nz=10)
    space = LagrangeFESpace(mesh, p=p)
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()
    NF = mesh.number_of_faces()
    NC = mesh.number_of_cells()
    assert space.geo_dimension() == 3
    assert space.top_dimension() == 3
    assert space.number_of_global_dofs() == NN + NE*(p-1) + NF*(p-2)*(p-1)//2 + NC*(p-3)*(p-2)*(p-1)//6
    assert space.number_of_local_dofs() == (p+1)*(p+2)*(p+3)//6

    mesh = TetrahedronMesh.from_unit_sphere_gmsh(h=0.2)
    space = LagrangeFESpace(mesh, p=p)
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()
    NF = mesh.number_of_faces()
    NC = mesh.number_of_cells()
    assert space.geo_dimension() == 3
    assert space.top_dimension() == 3
    assert space.number_of_global_dofs() == NN + NE*(p-1) + NF*(p-2)*(p-1)//2 + NC*(p-3)*(p-2)*(p-1)//6
    assert space.number_of_local_dofs() == (p+1)*(p+2)*(p+3)//6

def test_interpolation_fe_function(p=1, method='rg'):
    import copy
    import time
    mesh = TriangleMesh.from_box([0, 2, 0, 1], nx=20, ny=10)
    mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3)
    #mesh = HalfEdgeMesh2d.from_box([0, 2, 0, 1], 0.3)
    mesho = copy.deepcopy(mesh)
    space = LagrangeFESpace(mesh, p=p) 
    spaceo = LagrangeFESpace(mesho, p=p) 

    def u(p): return np.sin(2*p[..., 1])*np.sin(2*p[..., 0])
    uIo = spaceo.interpolate(u)  

    err = mesh.error(u, uIo)
    print(err)

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
            space = LagrangeFESpace(mesh, p=p)

            t0 = time.time()
            uI = space.interpolation_fe_function(uIo)
            t1 = time.time()
            print("插值用时: ", t1-t0, "s")

            mesho = copy.deepcopy(mesh)
            spaceo = LagrangeFESpace(mesho, p=p)
            uIo = spaceo.function()
            uIo[:] = uI[:]

            mesh1 = copy.deepcopy(mesh)
            node = mesh1.entity('node')
            node = np.c_[node, uI[:, None]]
            mesh1.node = node
            mesh1.to_vtk(fname='out_ref_'+str(fff).zfill(3)+".vtu")

            err0 = mesh.error(u, uIo)
            print("err0 : ", err0)
            print(mesho.number_of_cells())

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
            space = LagrangeFESpace(mesh, p=p)

            t0 = time.time()
            uI = space.interpolation_fe_function(uIo)
            t1 = time.time()
            print("插值用时: ", t1-t0, "s")

            mesho = copy.deepcopy(mesh)
            spaceo = LagrangeFESpace(mesho, p=p)
            uIo = spaceo.function()
            uIo[:] = uI[:]

            mesh1 = copy.deepcopy(mesh)
            node = mesh1.entity('node')
            node = np.c_[node, uI[:, None]]
            mesh1.node = node
            mesh1.to_vtk(fname='out_ref_'+str(fff).zfill(3)+".vtu")

            err0 = mesh.error(u, uIo)
            print("err0 : ", err0)
            print(mesho.number_of_cells())

def test_domain_with_hole():
    from fealpy.geometry import SquareWithCircleHoleDomain
    domain = SquareWithCircleHoleDomain() 
    mesh = TriangleMesh.from_domain_distmesh(domain, 0.05, maxit=100)
    mesh = HalfEdgeMesh2d.from_mesh(mesh, NV=3) # 使用半边网格

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    mesh.find_cell(axes)
    #hmesh.add_halfedge_plot(axes, showindex=True)
    #axes.scatter(points[:, 0], points[:, 1], color='r')
    #for i in range(len(points)):
    #    plt.annotate(i, points[i], textcoords="offset points", xytext=(0, 10),
    #            ha='center', color='r', fontsize=40)
    plt.show()





if __name__ == '__main__':
    #test_triangle_mesh(2)
    #test_interpolation_fe_function(p=1, method='nvb')
    test_domain_with_hole()

