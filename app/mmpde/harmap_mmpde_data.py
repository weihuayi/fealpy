from fealpy.backend import backend_manager as bm
from fealpy.geometry import RectangleDomain
from fealpy.mesh import TriangleMesh
from fealpy.mesh import TriangleMesh as TM


h = 0.033
domain = RectangleDomain(hmin=h)
vertices = bm.array([[0,0],[1,0],[1,0.5],[0.5,0.5],[0.5,1],[0,1]])
mesh1 = TM.from_domain_distmesh(domain ,maxit=100)
mesh2 = TM.from_polygon_gmsh(vertices, h)
def thr(p):
    x = p[...,0]
    y=  p[...,1]
    area = bm.array([0.5,1,0.5,1,-0.4,0.4])
    in_x = (x >= area[0]) & (x <= area[1])
    in_y = (y >= area[2]) & (y <= area[3])
    if p.shape[-1] == 3:
        z = p[...,2]
        in_z = (z >= area[4]) & (z <= area[5])
        return in_x & in_y & in_z
    return  in_x & in_y

mesh_data = {
    'from_box':TriangleMesh.from_box([0,1,0,1],nx=40,ny=40),
    'from_box_Lshape':TriangleMesh.from_box([0,1,0,1],nx=40,ny=40,threshold=thr),
    'from_domain_distmesh':TriangleMesh(mesh1.node,mesh1.ds.cell),
    'from_polygon_gmsh_Lshape':TriangleMesh(mesh2.node,mesh2.ds.cell),
}
function_data = {
    'u1':'(1-exp(4*(x-1))*sin(pi*y))',
    'u2':'1/2 + 1/2 * tanh(100*(1/12 - (x-1/2)**2 -(y-1/2)**2))',
    'u3':'exp(-100*((x-0.5)**2 + (y-0.5)**2))',
    'u4':'1/(1+ exp(100*(x+y-1)))'
}