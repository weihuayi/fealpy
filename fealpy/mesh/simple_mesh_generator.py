
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



def squaremesh(x0, x1, y0, y1, r=3, dtype=np.float):
    nodes = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=dtype)
    cells = np.array([[1, 2, 0], [3, 0, 2]], dtype=np.int)
    mesh = TriangleMesh(nodes, cells, dtype=dtype)
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


