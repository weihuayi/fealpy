
from ..backend import bm
from ..mesh import TriangleMesh


def indofP1(mesh: TriangleMesh, threshold=None, return_index=False, tensor_mesh=False):
    isDDof = mesh.boundary_node_flag()
    points = mesh.entity('node')
    if tensor_mesh:
        points = bm.concat([points, bm.zeros((len(points),1), dtype=bm.float64)], axis=1)

    index_dof = bm.arange(len(points))[isDDof]
    bd_point = points[isDDof] 
    flag = threshold(bd_point)
    index_dof = index_dof[flag]

    bd_flag = bm.zeros((len(points),), dtype=bm.bool)
    bm.set_at(bd_flag, index_dof, True)
    if return_index:
        return ~bd_flag, index_dof
    return ~bd_flag


def transferP1red(mesh: TriangleMesh, level:int, threshold=None, tensor_mesh=False):
    Pro_p = [None]*(level-1)
    
    if threshold == None:
            return mesh.uniform_refine(n=level-1, returnim=True)[::-1]

    for i in range(level-1):
        flag0 = indofP1(mesh, threshold=threshold, tensor_mesh=tensor_mesh)
        P = mesh.uniform_refine(n=1, returnim=True)[0]
        flag1 = indofP1(mesh, threshold=threshold, tensor_mesh=tensor_mesh)
        Pro_p[i] = P.to_scipy()[bm.to_numpy(flag1)][:,bm.to_numpy(flag0)]

    return Pro_p
