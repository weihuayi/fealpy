import numpy as np
import matplotlib.pyplot as plt
import pytest
import copy
from fealpy.mesh import TetrahedronMesh 
from fealpy.functionspace import LagrangeFESpace
import ipdb

@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_uniform_refine(n):
    mesh = TetrahedronMesh.from_one_tetrahedron(meshtype='equ')
    mesh.uniform_refine(n=n)

    vol = mesh.entity_measure('cell')
    assert np.all(vol>0)


@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_interpolate(p):
    mesh = TetrahedronMesh.from_one_tetrahedron(meshtype='equ')

    mesh.uniform_refine(n=3)

    ips0 = mesh.interpolation_points(p)

    space = LagrangeFESpace(mesh, p=p)
    ips1 = space.interpolation_points()

    assert np.allclose(ips0, ips1)

    c2d0 = mesh.cell_to_ipoint(p)
    c2d1 = space.cell_to_dof()

    assert np.all(c2d0 == c2d1)

def test_mesh_generation_on_cylinder_by_gmsh():
    mesh = TetrahedronMesh.from_cylinder_gmsh(1, 5, 0.1)
    mesh.add_plot(plt)
    plt.show()

def test_mesh_generation_by_meshpy():
    points = np.array([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        ])

    facets = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ])
    
    mesh = TetrahedronMesh.from_meshpy(points, facets, 0.2)
    mesh.add_plot(plt)
    plt.show()

def tet_volume(node):
    v01 = node[1] - node[0]
    v02 = node[2] - node[0]
    v03 = node[3] - node[0]
    return np.sum(np.cross(v01, v02)*v03)/6

def is_in_the_tetrahedron(point, node):
    node = np.r_[node, point[None, :]]
    localCell = np.array([[0, 1, 2, 4], 
                          [1, 3, 2, 4], 
                          [0, 3, 1, 4], 
                          [2, 3, 0, 4]], dtype=np.int_)
    for i in range(4):
        tau = localCell[i]
        v = tet_volume(node[tau])
        if v < 0:
            print(i)
            return False
    return True

def test_bisect():
    mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], 5, 5, 5)
    NC = mesh.number_of_cells()
    isMarkedCell = np.zeros(NC, dtype=np.bool_) 
    isMarkedCell[[1, 10, 14, 26, 24, 12, 100, 50]] = True

    mesh_copy = copy.deepcopy(mesh)
    cell_copy = mesh_copy.entity('cell')
    node_copy = mesh_copy.entity('node')

    options = mesh.bisect_options(HB=True)
    mesh.bisect(isMarkedCell, options=options)
    NC = mesh.number_of_cells()

    cbar = mesh.entity_barycenter("cell")
    HB = options['HB']
    for i in range(NC):
        c0 = HB[i, 0]
        c1 = HB[i, 1]
        p = cbar[c0]
        n = node_copy[cell_copy[c1]]
        flag = is_in_the_tetrahedron(p, n)
        if(not flag):
            print("ERROR", i, c0, c1)

def dis(p):
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]
    val = np.sin(x)*np.sin(y)*np.sin(z)
    return val

def test_interplation_weith_HB():

    mesh = TetrahedronMesh.from_unit_cube()
    space = LagrangeFESpace(mesh, p=1)

    volume = mesh.entity_measure()

    NC = mesh.number_of_cells()
    
    u0 = space.interpolate(dis)
    H = np.zeros(NC, dtype=np.float64)
    H[:] = dis(mesh.entity_barycenter("cell"))
    mesh.celldata['H'] = H.copy()
    mesh.to_vtk(fname = 'aaa.vtu')

    error0 = mesh.error(u0, dis) 
    print('连续函数的插值误差 : ', error0)

    cell2dof = mesh.cell_to_ipoint(p=1)
 
    isMarkedCell = np.abs(np.sum(u0[cell2dof], axis=-1)) > 1
    data = {'nodedata': [u0], 'celldata': [H]}
    options = mesh.bisect_options(data=data, HB=True)
    mesh.bisect(isMarkedCell, options=options)
    
    data = options['data']
    fval = data['nodedata'][0]
    fval = space.function(array=fval)

    H = data["celldata"][0]
    error_H = H - dis(mesh.entity_barycenter("cell"))
    #error_H = mesh.celldata['H'] - dis(mesh.entity_barycenter("cell"))
    print("??? : ", H - mesh.celldata['H'])
    print(H)
    print(error_H)
    print(np.max(np.abs(error_H)))

    error_u0 = fval - dis(mesh.entity("node"))
    print(np.max(np.abs(error_u0)))

    error = mesh.error(fval, dis)
    print('error:', error)
    mesh.to_vtk(fname='bbb.vtu')

if __name__ == "__main__":
    #test_mesh_generation_by_meshpy()
    #test_bisect()
    test_interplation_weith_HB()
