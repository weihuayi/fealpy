
import numpy as np
from .TriangleMesh import TriangleMesh, TriangleMeshWithInfinityNode
from .QuadrangleMesh import QuadrangleMesh 
from .HexahedronMesh import HexahedronMesh 
from .PolygonMesh import PolygonMesh 

from .level_set_function import DistDomain2d, DistDomain3d
from .level_set_function import dcircle, drectangle
from .level_set_function import ddiff 
from .sizing_function import huniform
from .distmesh import DistMesh2d 



def squaremesh(x0, x1, y0, y1, r=3):
    nodes = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float)
    cells = np.array([[1, 2, 0], [3, 0, 2]], dtype=np.int)
    mesh = TriangleMesh(nodes, cells)
    mesh.uniform_refine(r)
    return mesh 

def rectangledomainmesh(box, nx=10, ny=10, meshtype='tri'):
    N = (nx+1)*(ny+1)
    NC = nx*ny
    node = np.zeros((N,2))
    X, Y = np.mgrid[box[0]:box[1]:complex(0,nx+1), box[2]:box[3]:complex(0,ny+1)]
    node[:,0] = X.flatten()
    node[:,1] = Y.flatten()

    idx = np.arange(N).reshape(nx+1, ny+1)
    if meshtype=='tri':
        cell = np.zeros((2*NC, 3), dtype=np.int)
        cell[:NC, 0] = idx[1:,0:-1].flatten(order='F')
        cell[:NC, 1] = idx[1:,1:].flatten(order='F')
        cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 2] = idx[1:, 1:].flatten(order='F')
        return TriangleMesh(node, cell)
    elif meshtype == 'quad':
        cell = np.zeros((NC,4), dtype=np.int)
        cell[:,0] = idx[0:-1, 0:-1].flatten()
        cell[:,1] = idx[1:, 0:-1].flatten()
        cell[:,2] = idx[1:, 1:].flatten()
        cell[:,3] = idx[0:-1, 1:].flatten()
        return QuadrangleMesh(node, cell)
    elif meshtype == 'polygon':
        cell = np.zeros((NC,4), dtype=np.int)
        cell[:,0] = idx[0:-1, 0:-1].flatten()
        cell[:,1] = idx[1:, 0:-1].flatten()
        cell[:,2] = idx[1:, 1:].flatten()
        cell[:,3] = idx[0:-1, 1:].flatten()
        return PolygonMesh(node, cell)

def triangle(box, h, meshtype='tri'):
    from meshpy.triangle import MeshInfo, build
    mesh_info = MeshInfo()
    mesh_info.set_points([(box[0], box[2]), (box[1], box[2]), (box[1], box[3]), (box[0], box[3])])
    mesh_info.set_facets([[0,1], [1,2], [2,3], [3,0]])  
    mesh = build(mesh_info, max_volume=h**2)
    node = np.array(mesh.points, dtype=np.float)
    cell = np.array(mesh.elements, dtype=np.int)
    if meshtype is 'tri':
        return TriangleMesh(node, cell)
    elif meshtype is 'polygon':
        mesh = TriangleMeshWithInfinityNode(TriangleMesh(node, cell))
        pnode, pcell, pcellLocation = mesh.to_polygonmesh()
        return PolygonMesh(pnode, pcell, pcellLocation) 

def distmesh2d(fd, h0, bbox, pfix, meshtype='tri', dtype=np.float):
    fh = huniform
    domain = DistDomain2d(fd, fh, bbox, pfix)
    distmesh2d = DistMesh2d(domain, h0)
    distmesh2d.run()
    if meshtype is 'tri':
        return distmesh2d.mesh
    elif meshtype is 'polygon':
        mesh = TriangleMeshWithInfinityNode(distmesh2d.mesh)
        pnode, pcell, pcellLocation = mesh.to_polygonmesh()
        return PolygonMesh(pnode, pcell, pcellLocation) 

def unitcircledomainmesh(h0, meshtype='tri', dtype=np.float):
    fd = lambda p: dcircle(p,(0,0),1)
    fh = huniform
    bbox = [-1.2, 1.2, -1.2, 1.2]
    pfix = None 
    domain = DistDomain2d(fd, fh, bbox, pfix)
    distmesh2d = DistMesh2d(domain, h0)
    distmesh2d.run()
    if meshtype is 'tri':
        return distmesh2d.mesh
    elif meshtype is 'polygon':
        mesh = TriangleMeshWithInfinityNode(distmesh2d.mesh)
        pnode, pcell, pcellLocation = mesh.to_polygonmesh()
        return PolygonMesh(pnode, pcell, pcellLocation) 

def cubehexmesh(cube, nx=10, ny=10, nz=10):
    N = (nx+1)*(ny+1)*(nz+1)
    NC = nx*ny*nz
    node = np.zeros((N, 3), dtype=np.float)
    X, Y, Z = np.mgrid[
            cube[0]:cube[1]:complex(0, nx+1), 
            cube[2]:cube[3]:complex(0, ny+1),
            cube[4]:cube[5]:complex(0, nz+1)
            ]
    node[:, 0] = X.flatten()
    node[:, 1] = Y.flatten()
    node[:, 2] = Z.flatten()

    idx = np.arange(N).reshape(nx+1, ny+1, nz+1)
    c = idx[:-1, :-1, :-1]

    cell = np.zeros((NC, 8), dtype=np.int)
    nyz = (ny + 1)*(nz + 1)
    cell[:, 0] = c.flatten()
    cell[:, 1] = cell[:, 0] + nyz
    cell[:, 2] = cell[:, 1] + nz + 1
    cell[:, 3] = cell[:, 0] + nz + 1
    cell[:, 4] = cell[:, 0] + 1
    cell[:, 5] = cell[:, 4] + nyz
    cell[:, 6] = cell[:, 5] + nz + 1
    cell[:, 7] = cell[:, 4] + nz + 1

    return HexahedronMesh(node, cell)


def fishbone(box, n=10):
    qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
    node = qmesh.entity('node')
    cell = qmesh.entity('cell')
    NC = qmesh.number_of_cells()
    isLeftCell = np.zeros((n, n), dtype=np.bool)
    isLeftCell[0::2, :] = True
    isLeftCell = isLeftCell.reshape(-1)
    lcell = cell[isLeftCell]
    rcell = cell[~isLeftCell]
    newCell = np.r_['0', 
            lcell[:, [1, 2, 0]], 
            lcell[:, [3, 0, 2]],
            rcell[:, [0, 1, 3]],
            rcell[:, [2, 3, 1]]]
    return TriangleMesh(node, newCell)

def cross_mesh(box, n=10):
    qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
    node = qmesh.entity('node')
    cell = qmesh.entity('cell')
    NN = qmesh.number_of_nodes()
    NC = qmesh.number_of_cells()
    bc = qmesh.barycenter('cell') 
    newNode = np.r_['0', node, bc]

    newCell = np.zeros((4*NC, 3), dtype=np.int) 
    newCell[0:NC, 0] = range(NN, NN+NC)
    newCell[0:NC, 1:3] = cell[:, 0:2]

    newCell[NC:2*NC, 0] = range(NN, NN+NC)
    newCell[NC:2*NC, 1:3] = cell[:, 1:3]

    newCell[2*NC:3*NC, 0] = range(NN, NN+NC)
    newCell[2*NC:3*NC, 1:3] = cell[:, 2:4]

    newCell[3*NC:4*NC, 0] = range(NN, NN+NC)
    newCell[3*NC:4*NC, 1:3] = cell[:, [3, 0]] 
    return TriangleMesh(newNode, newCell)

def rice_mesh(box, n=10):
    qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
    node = qmesh.entity('node')
    cell = qmesh.entity('cell')
    NC = qmesh.number_of_cells()

    isLeftCell = np.zeros((n, n), dtype=np.bool)
    isLeftCell[0, 0::2] = True
    isLeftCell[1, 1::2] = True
    if n > 2:
        isLeftCell[2::2, :] = isLeftCell[0, :]
    if n > 3:
        isLeftCell[3::2, :] = isLeftCell[1, :]
    isLeftCell = isLeftCell.reshape(-1)

    lcell = cell[isLeftCell]
    rcell = cell[~isLeftCell]
    newCell = np.r_['0', 
            lcell[:, [1, 2, 0]], 
            lcell[:, [3, 0, 2]],
            rcell[:, [0, 1, 3]],
            rcell[:, [2, 3, 1]]]
    return TriangleMesh(node, newCell)


def nonuniform_mesh(box, n=10):
    nx = 4*n
    ny = 4*n
    n1 = 2*n+1

    N = n1**2
    NC = 4*n*n
    node = np.zeros((N,2))

    x = np.zeros(n1, dtype=np.float)
    x[0::2] = range(0,nx+1,4)
    x[1::2] = range(3, nx+1, 4)

    y = np.zeros(n1, dtype=np.float)
    y[0::2] = range(0,nx+1,4)
    y[1::2] = range(1, nx+1,4)

    node[:,0] = x.repeat(n1)/nx
    node[:,1] = np.tile(y, n1)/ny


    idx = np.arange(N).reshape(n1, n1)

    cell = np.zeros((2*NC, 3), dtype=np.int)
    cell[:NC, 0] = idx[1:,0:-1].flatten(order='F')
    cell[:NC, 1] = idx[1:,1:].flatten(order='F')
    cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
    cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
    cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
    cell[NC:, 2] = idx[1:, 1:].flatten(order='F')
    return TriangleMesh(node, cell)
        
def uncross_mesh(box, n=10, r="1"):
    qmesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='quad')
    node = qmesh.entity('node')
    cell = qmesh.entity('cell')
    NN = qmesh.number_of_nodes()
    NC = qmesh.number_of_cells()   
    bc = qmesh.barycenter('cell') 

    if r=="1":
        bc1 = np.sqrt(np.sum((bc-node[cell[:,0], :])**2, axis=1))[0]
        newNode = np.r_['0', node, bc-bc1*0.3]
    elif r=="2":
        ll = node[cell[:, 0]] - node[cell[:, 2]]
        bc = qmesh.barycenter('cell') + ll/4
        newNode = np.r_['0',node, bc]

    newCell = np.zeros((4*NC, 3), dtype=np.int) 
    newCell[0:NC, 0] = range(NN, NN+NC)
    newCell[0:NC, 1:3] = cell[:, 0:2]
        
    newCell[NC:2*NC, 0] = range(NN, NN+NC)
    newCell[NC:2*NC, 1:3] = cell[:, 1:3]

    newCell[2*NC:3*NC, 0] = range(NN, NN+NC)
    newCell[2*NC:3*NC, 1:3] = cell[:, 2:4]

    newCell[3*NC:4*NC, 0] = range(NN, NN+NC)
    newCell[3*NC:4*NC, 1:3] = cell[:, [3, 0]] 
    return TriangleMesh(newNode, newCell)

