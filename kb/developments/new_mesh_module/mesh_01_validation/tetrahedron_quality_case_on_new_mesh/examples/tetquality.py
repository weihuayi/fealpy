import argparse

from fealpy.backend import bm
from fealpy.mesh import Mesh, MeshBlock, EntitySector, TopologyBuilder

def from_vtu(file):
    import meshio
    data = meshio.read(file)
    node = data.points
    cell = data.cells_dict["tetra"]
    cell = cell.astype(bm.int32)
    storage = MeshBlock(positions=node)
    storage.add_sector(EntitySector("tet",cell),root=True)
    TopologyBuilder.construct(storage)
    mesh = Mesh(storage)
    return mesh

def radius_ratio(mesh):
    s = face_area(mesh)
    cell2face = mesh.sector('tet').to('tri').tgt_indices
    ss = bm.sum(s[cell2face],axis=1)
    node = mesh.block.positions
    cell = mesh.sector("tet").indices
    v10 = node[cell[:, 0],:] - node[cell[:, 1],:]
    v20 = node[cell[:, 0],:] - node[cell[:, 2],:]
    v30 = node[cell[:, 0],:] - node[cell[:, 3],:]
    l10 = bm.sum(v10**2, axis=1, keepdims=True)
    l20 = bm.sum(v20**2, axis=1, keepdims=True)
    l30 = bm.sum(v30**2, axis=1, keepdims=True)
    d = l10*bm.cross(v20, v30) + l20*bm.cross(v30, v10) + l30*bm.cross(v10, v20)
    ld = bm.sqrt(bm.sum(d**2,axis=1))
    vol = cell_volume(mesh)
    R = ld/vol/12.0
    r = 3.0*vol/ss
    return R/r/3.0

def dihedral_angle(mesh):
    """
    @brief 计算所有单元的四个二面角
    """
    from fealpy.mesh import TetrahedronSchema
    node = mesh.block.positions
    cell = mesh.sector("tet").indices
    localFace = TetrahedronSchema.local_entity('tri')

    n = [bm.cross(node[cell[:, j],:] - node[cell[:, i],:],
        node[cell[:, k],:] - node[cell[:, i],:]) for i, j, k in localFace]
    l =[bm.sqrt(bm.sum(ni**2, axis=1)) for ni in n]
    n = [ni/li.reshape(-1, 1) for ni, li in zip(n, l)]
    localEdge = TetrahedronSchema.local_entity('edge')
    angle = [(bm.pi - bm.arccos((n[i]*n[j]).sum(axis=1)))/bm.pi*180 for i,j in localEdge[-1::-1]]
    return bm.array(angle).T

def face_area(mesh):
    face = mesh.sector("tri").indices
    node = mesh.block.positions
    v01 = node[face[:, 1],:] - node[face[:, 0],:]
    v02 = node[face[:, 2],:] - node[face[:, 0],:]
    nv = bm.cross(v01, v02)
    area = bm.sqrt(bm.square(nv).sum(axis=1))/2.0
    return area

def cell_volume(mesh):
    cell = mesh.sector("tet").indices
    node = mesh.block.positions
    v01 = node[cell[:, 1],:] - node[cell[:, 0],:]
    v02 = node[cell[:, 2],:] - node[cell[:, 0],:]
    v03 = node[cell[:, 3],:] - node[cell[:, 0],:]
    volume = bm.abs(bm.sum(v03*bm.cross(v01, v02), axis=1))/6.0
    return volume

#filename = 'data/sphere_init.vtu'
parser = argparse.ArgumentParser(description='test mesh quality')
parser.add_argument('--filename', type=str, default=None,
                    help='the vtu file')
args = parser.parse_args()
filename = args.filename
if filename is None:
    raise ValueError("Please provide the vtu file using --filename argument.")

mesh = from_vtu(filename)
dihedral = dihedral_angle(mesh)
print(dihedral)
radius = radius_ratio(mesh)
print(radius)
