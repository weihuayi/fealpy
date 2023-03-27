import numpy as np

from .IntervalMesh import IntervalMesh
from .TriangleMesh import TriangleMesh, TriangleMeshWithInfinityNode
from .QuadrangleMesh import QuadrangleMesh
from .TetrahedronMesh import TetrahedronMesh
from .HexahedronMesh import HexahedronMesh
from .PolygonMesh import PolygonMesh
from .HalfEdgeMesh2d import HalfEdgeMesh2d
from .StructureQuadMesh import StructureQuadMesh

from fealpy.functionspace import LagrangeFiniteElementSpace

from ..geometry import DistDomain2d, DistDomain3d
from ..geometry import dcircle, drectangle
from ..geometry import ddiff
from ..geometry import huniform
from ..decorator import timer

from .LagrangeQuadrangleMesh import LagrangeQuadrangleMesh
from .LagrangeTriangleMesh import LagrangeTriangleMesh


from .distmesh import DistMesh2d

#from .interface_mesh_generator import InterfaceMesh2d

def write_to_vtu(fname, mesh, nodedata=None, celldata=None, p=1):
    if p == 1:
        if nodedata is not None:
            for key, val in nodedata.items():
                mesh.nodedata[key] = val
        if celldata is not None:
            for key, val in celldata.items():
                mesh.celldata[key] = val
        mesh.to_vtk(fname=fname)
    else:
        space = LagrangeFiniteElementSpace(mesh, p=p)
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        lmesh = LagrangeTriangleMesh(node, cell, p=p)
        lmesh.node = space.interpolation_points() 
        lmesh.ds.cell = space.cell_to_dof()
        if nodedata is not None:
            for key, val in nodedata.items():
                lmesh.nodedata[key] = val
        if celldata is not None:
            for key, val in celldata.items():
                lmesh.celldata[key] = val
        lmesh.to_vtk(fname=fname)

def circle_interval_mesh(c, r, h, clock = 'w'):

    n = int(2*np.pi*r/h)
    dt = 2*np.pi/n
    theta  = np.arange(0, 2*np.pi, dt)

    node = np.zeros((n, 2),dtype = np.float64)
    cell = np.zeros((n, 2),dtype = np.int_)

    if clock == 'w':
        node[:, 0] = r*np.cos(theta)
        node[:, 1] = -r*np.sin(theta)

    if clock == 'cw':
        node[:, 0] = r*np.cos(theta)
        node[:, 1] = r*np.sin(theta)

    node[:,0] = node[:,0] + c[0]
    node[:,1] = node[:,1] + c[1]

    cell[:,0] = np.arange(n)
    cell[:,1][:-1] = np.arange(1,n)

    return node, cell

def meshpy2d(points, facets, h, hole_points=None, facet_markers=None, point_markers=None, meshtype='tri'):
    """
    调用 meshpy 生成二维三角形或多边形网格
    """
    from meshpy.triangle import MeshInfo, build

    mesh_info = MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets)

    if hole_points is not None:
        mesh_info.set_holes(hole_points)

    mesh = build(mesh_info, max_volume=h**2)

    node = np.array(mesh.points, dtype=np.float64)
    cell = np.array(mesh.elements, dtype=np.int_)

    if meshtype in {'t', 'tri', 'triangle'}:
        return TriangleMesh(node, cell)
    elif meshtype in {'p', 'polygon', 'poly'}:
        mesh = TriangleMeshWithInfinityNode(TriangleMesh(node, cell))
        pnode, pcell, pcellLocation = mesh.to_polygonmesh()
        return PolygonMesh(pnode, pcell, pcellLocation)

def split_mesh(mesh, entity='cell'):
    from fealpy.graph import metis
    edgecuts, parts = metis.part_mesh(tmesh, nparts=n, entity=entity)
    return edgecuts, parts

def delete_cell(node, cell, threshold):
    """

    Notes
    -----
    利用 threshhold 来删除一部分网格单元。threshold 以单元的重心为输入参数，
    返回一个逻辑数组，需要删除的单元标记为真。
    """
    NN = len(node)
    bc = np.sum(node[cell, :], axis=1)/cell.shape[1]
    isDelCell = threshold(bc) 
    cell = cell[~isDelCell]
    isValidNode = np.zeros(NN, dtype=np.bool_)
    isValidNode[cell] = True
    node = node[isValidNode]
    idxMap = np.zeros(NN, dtype=cell.dtype)
    idxMap[isValidNode] = range(isValidNode.sum())
    cell = idxMap[cell]

    return node, cell

def one_triangle_mesh(meshtype='iso'):
    if meshtype == 'equ':
        node = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2]], dtype=np.float64)
    elif meshtype == 'iso':
        node = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]], dtype=np.float64)
    cell = np.array([[0, 1, 2]], dtype=np.int_)
    return TriangleMesh(node, cell)

def one_quad_mesh(meshtype='square'):
    if meshtype in {'square', 'zhengfangxing'}:
        node = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]], dtype=np.float64)
    elif meshtype in {'rectangle', 'rec', 'juxing'}:
        node = np.array([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0]], dtype=np.float64)
    elif meshtype in {'rhombus', 'lingxing'}:
        node = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.5, np.sqrt(3)/2],
            [0.5, np.sqrt(3)/2]], dtype=np.float64)
    cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
    return QuadrangleMesh(node, cell)

def one_tetrahedron_mesh(meshtype='equ'):
    if meshtype == 'equ':
        node = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3)/2, 0.0],
            [0.5, np.sqrt(3)/6, np.sqrt(2/3)]], dtype=np.float64)
    elif meshtype == 'iso':
        node = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]], dtype=np.float64)
    cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
    return TetrahedronMesh(node, cell)

def interval_mesh(interval=[0, 1], nx=1):
    node = np.linspace(interval[0], interval[1], nx+1, dtype=np.float64).reshape(-1, 1)
    cell = np.zeros((nx, 2), dtype=np.int_)
    cell[:, 0] = np.arange(0, nx)
    cell[:, 1] = np.arange(1, nx+1)
    return IntervalMesh(node, cell)


def boxmesh2d(box, nx=10, ny=10, meshtype='tri', threshold=None,
        returnnc=False, p=None):
    """

    Notes
    -----
    生成二维矩形区域上的网格，包括结构的三角形、四边形和三角形对偶的多边形网
    格. 
    """
    N = (nx+1)*(ny+1)
    NC = nx*ny
    node = np.zeros((N,2))
    X, Y = np.mgrid[box[0]:box[1]:complex(0,nx+1), box[2]:box[3]:complex(0,ny+1)]
    node[:, 0] = X.flatten()
    node[:, 1] = Y.flatten()
    NN = len(node)
    print(N)

    idx = np.arange(N).reshape(nx+1, ny+1)
    if meshtype in {'tri', 'triangle'}:
        cell = np.zeros((2*NC, 3), dtype=np.int_)
        cell[:NC, 0] = idx[1:,0:-1].flatten(order='F')
        cell[:NC, 1] = idx[1:,1:].flatten(order='F')
        cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 2] = idx[1:, 1:].flatten(order='F')

        if threshold is not None:
            node, cell = delete_cell(node, cell, threshold)

        if returnnc:
            return node, cell
        elif p is None:
            return TriangleMesh(node, cell)
        else:
            return LagrangeTriangleMesh(node, cell, p=p)
    elif meshtype == 'quad':
        cell = np.zeros((NC,4), dtype=np.int_)
        cell[:,0] = idx[0:-1, 0:-1].flat
        cell[:,1] = idx[1:, 0:-1].flat
        cell[:,2] = idx[1:, 1:].flat
        cell[:,3] = idx[0:-1, 1:].flat
        if threshold is not None:
            node, cell = delete_cell(node, cell, threshold)
        if returnnc:
            return node, cell
        elif p is None:
            return QuadrangleMesh(node, cell)
        else:
            return LagrangeQuadrangleMesh(node, cell[:, [0, 3, 1, 2]], p=p)
    elif meshtype in {'polygon', 'poly'}:
        cell = np.zeros((2*NC, 3), dtype=np.int_)
        cell[:NC, 0] = idx[1:,0:-1].flatten(order='F')
        cell[:NC, 1] = idx[1:,1:].flatten(order='F')
        cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 2] = idx[1:, 1:].flatten(order='F')

        if threshold is not None:
            node, cell = delete_cell(node, cell, threshold)

        mesh = TriangleMeshWithInfinityNode(TriangleMesh(node, cell))
        pnode, pcell, pcellLocation = mesh.to_polygonmesh()
        return PolygonMesh(pnode, pcell, pcellLocation)
    elif meshtype == 'noconvex':
        mesh0 = StructureQuadMesh(box, nx, ny)
        node0 = mesh0.entity("node")
        cell0 = mesh0.entity("cell")[:, [0, 2, 3, 1]]
        mesh = QuadrangleMesh(node0, cell0)

        edge = mesh.entity("edge")
        node = mesh.entity("node")
        cell = mesh.entity("cell")
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        cell2edge = mesh.ds.cell_to_edge()
        isbdedge = mesh.ds.boundary_edge_flag() 
        isbdcell = mesh.ds.boundary_cell_flag() 

        nie = np.sum(~isbdedge)
        hx = 1/nx
        hy = 1/ny
        newnode = np.zeros((NN+nie, 2), dtype=np.float_)
        newnode[:NN] = node
        newnode[NN:] = 0.5*node[edge[~isbdedge, 0]] + 0.5*node[edge[~isbdedge, 1]]
        newnode[NN: NN+(nx-1)*ny] = newnode[NN: NN+(nx-1)*ny] + np.array([[0.2*hx, 0.1*hy]])
        newnode[NN+(nx-1)*ny:] = newnode[NN+(nx-1)*ny:] + np.array([[0.1*hx, 0.2*hy]])

        edge2newnode = -np.ones(NE, dtype=np.int_)
        edge2newnode[~isbdedge] = np.arange(NN, newnode.shape[0])
        newcell = np.zeros((NC, 8), dtype=np.int_)
        newcell[:, ::2] = cell
        newcell[:, 1::2] = edge2newnode[cell2edge]

        flag = newcell>-1
        num = np.zeros(NC+1, dtype=np.int_)
        num[1:] = np.sum(flag, axis=-1)
        newcell = newcell[flag]
        cellLocation = np.cumsum(num)
        return PolygonMesh(newnode, newcell, cellLocation)


def boxmesh3d(box, nx=10, ny=10, nz=10, meshtype='hex', threshold=None,
        returnnc=False):
    """
    Notes
    -----
    生成长方体区域上的六面体或四面体网格。
    """
    N = (nx+1)*(ny+1)*(nz+1)
    NC = nx*ny*nz
    node = np.zeros((N, 3), dtype=np.float64)
    X, Y, Z = np.mgrid[
            box[0]:box[1]:complex(0, nx+1), 
            box[2]:box[3]:complex(0, ny+1),
            box[4]:box[5]:complex(0, nz+1)
            ]
    node[:, 0] = X.flatten()
    node[:, 1] = Y.flatten()
    node[:, 2] = Z.flatten()

    idx = np.arange(N).reshape(nx+1, ny+1, nz+1)
    c = idx[:-1, :-1, :-1]

    cell = np.zeros((NC, 8), dtype=np.int_)
    nyz = (ny + 1)*(nz + 1)
    cell[:, 0] = c.flatten()
    cell[:, 1] = cell[:, 0] + nyz
    cell[:, 2] = cell[:, 1] + nz + 1
    cell[:, 3] = cell[:, 0] + nz + 1
    cell[:, 4] = cell[:, 0] + 1
    cell[:, 5] = cell[:, 4] + nyz
    cell[:, 6] = cell[:, 5] + nz + 1
    cell[:, 7] = cell[:, 4] + nz + 1
    if meshtype == 'hex':
        if threshold is not None:
            node, cell = delete_cell(node, cell, threshold)
        if returnnc:
            return node, cell
        else:
            return HexahedronMesh(node, cell)
    elif meshtype == 'tet':
        localCell = np.array([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=np.int_)
        cell = cell[:, localCell].reshape(-1, 4)
        if threshold is not None:
            node, cell = delete_cell(node, cell, threshold)
        if returnnc:
            return node, cell
        else:
            return TetrahedronMesh(node, cell)

def triangle(box, h, meshtype='tri'):
    """
    Notes
    -----
    生成矩形区域上的非结构网格网格
    """
    from meshpy.triangle import MeshInfo, build
    mesh_info = MeshInfo()
    mesh_info.set_points([(box[0], box[2]), (box[1], box[2]), (box[1], box[3]), (box[0], box[3])])
    mesh_info.set_facets([[0, 1], [1, 2], [2, 3], [3, 0]])
    mesh = build(mesh_info, max_volume=h**2)
    node = np.array(mesh.points, dtype=np.float64)
    cell = np.array(mesh.elements, dtype=np.int_)
    if meshtype in {'tri', 'triangle'}:
        return TriangleMesh(node, cell)
    elif meshtype in {'polygon', 'poly'}:
        mesh = TriangleMeshWithInfinityNode(TriangleMesh(node, cell))
        pnode, pcell, pcellLocation = mesh.to_polygonmesh()
        return PolygonMesh(pnode, pcell, pcellLocation)

def special_boxmesh2d(box, n=10,
        meshtype='fishbone'):
    qmesh = boxmesh2d(box, nx=n, ny=n, meshtype='quad')
    node = qmesh.entity('node')
    cell = qmesh.entity('cell')
    NN = qmesh.number_of_nodes()
    NE = qmesh.number_of_edges()
    NC = qmesh.number_of_cells()
    if meshtype == 'fishbone':
        isLeftCell = np.zeros((n, n), dtype=np.bool_)
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
    elif meshtype == 'cross': 
        bc = qmesh.entity_barycenter('cell') 
        newNode = np.r_['0', node, bc]

        newCell = np.zeros((4*NC, 3), dtype=np.int_) 
        newCell[0:NC, 0] = range(NN, NN+NC)
        newCell[0:NC, 1:3] = cell[:, 0:2]
        
        newCell[NC:2*NC, 0] = range(NN, NN+NC)
        newCell[NC:2*NC, 1:3] = cell[:, 1:3]

        newCell[2*NC:3*NC, 0] = range(NN, NN+NC)
        newCell[2*NC:3*NC, 1:3] = cell[:, 2:4]

        newCell[3*NC:4*NC, 0] = range(NN, NN+NC)
        newCell[3*NC:4*NC, 1:3] = cell[:, [3, 0]] 
        return TriangleMesh(newNode, newCell)
    elif meshtype == 'rice':
        isLeftCell = np.zeros((n, n), dtype=np.bool_)
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
    elif meshtype == 'nonuniform':
        nx = 4*n
        ny = 4*n
        n1 = 2*n+1

        N = n1**2
        NC = 4*n*n
        node = np.zeros((N,2))
        
        x = np.zeros(n1, dtype=np.float64)
        x[0::2] = range(0,nx+1,4)
        x[1::2] = range(3, nx+1, 4)

        y = np.zeros(n1, dtype=np.float64)
        y[0::2] = range(0,nx+1,4)
        y[1::2] = range(1, nx+1,4)

        node[:,0] = x.repeat(n1)/nx
        node[:,1] = np.tile(y, n1)/ny


        idx = np.arange(N).reshape(n1, n1)
        
        cell = np.zeros((2*NC, 3), dtype=np.int_)
        cell[:NC, 0] = idx[1:,0:-1].flatten(order='F')
        cell[:NC, 1] = idx[1:,1:].flatten(order='F')
        cell[:NC, 2] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 0] = idx[0:-1, 1:].flatten(order='F')
        cell[NC:, 1] = idx[0:-1, 0:-1].flatten(order='F')
        cell[NC:, 2] = idx[1:, 1:].flatten(order='F')
        return TriangleMesh(node, cell)
        
def lshape_mesh(n=4):
    point = np.array([
        (-1, -1),
        (0, -1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1)], dtype=np.float64)

    cell = np.array([
        (1, 3, 0),
        (2, 0, 3),
        (3, 6, 2),
        (5, 2, 6),
        (4, 7, 3),
        (6, 3, 7)], dtype=np.int_)
    mesh = TriangleMesh(point, cell)
    mesh.uniform_refine(n)
    return mesh

def unitcirclemesh(h=0.1, meshtype='tri', p=None):
    """

    Reference
    ---------

    Notes
    -----
    利用 distmesh 算法生成单位圆上的非结构三角形或多边形网格
    """
    fd = lambda p: dcircle(p, (0,0), 1)
    fh = huniform
    bbox = [-1.2, 1.2, -1.2, 1.2]
    pfix = None 
    domain = DistDomain2d(fd, fh, bbox, pfix)
    distmesh2d = DistMesh2d(domain, h)
    distmesh2d.run()
    if meshtype in {'tri', 'triangle'}:
        if p is None:
            return distmesh2d.mesh
        else:
            node = distmesh2d.mesh.entity('node')
            cell = distmesh2d.mesh.entity('cell')
            mesh = LagrangeTriangleMesh(node, cell, p=p)
            return mesh
    elif meshtype in {'polygon', 'poly'}:
        mesh = TriangleMeshWithInfinityNode(distmesh2d.mesh)
        pnode, pcell, pcellLocation = mesh.to_polygonmesh()
        return PolygonMesh(pnode, pcell, pcellLocation) 

def distmesh2d(fd, fh, h0, bbox, pfix):
    domain = DistDomain2d(fd, fh, bbox, pfix)
    distmesh2d = DistMesh2d(domain, h0)
    distmesh2d.run()
    return distmesh2d.mesh

def polygon_mesh(meshtype='triquad'):
    if meshtype in {'triquad'}:
        node = np.array([
            (0.0, 0.0), (0.0, 1.0), (0.0, 2.0),
            (1.0, 0.0), (1.0, 1.0), (1.0, 2.0),
            (2.0, 0.0), (2.0, 1.0), (2.0, 2.0)], dtype=np.float64)
        cell = np.array([0, 3, 4, 4, 1, 0,
            1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5], dtype=np.int_)
        cellLocation = np.array([0, 3, 6, 10, 14, 18], dtype=np.int_)
        mesh = PolygonMesh(node, cell, cellLocation)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        return mesh

def interfacemesh2d(interface, n=20):
    alg = InterfaceMesh2d(interface, interface.box, n)
    ppoint, pcell, pcellLocation = alg.run()
    pmesh = PolygonMesh(ppoint, pcell, pcellLocation)
    return pmesh
       
# 下面的程序还需要标准化
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

    newCell = np.zeros((4*NC, 3), dtype=np.int_) 
    newCell[0:NC, 0] = range(NN, NN+NC)
    newCell[0:NC, 1:3] = cell[:, 0:2]
        
    newCell[NC:2*NC, 0] = range(NN, NN+NC)
    newCell[NC:2*NC, 1:3] = cell[:, 1:3]

    newCell[2*NC:3*NC, 0] = range(NN, NN+NC)
    newCell[2*NC:3*NC, 1:3] = cell[:, 2:4]

    newCell[3*NC:4*NC, 0] = range(NN, NN+NC)
    newCell[3*NC:4*NC, 1:3] = cell[:, [3, 0]] 
    return TriangleMesh(newNode, newCell)




