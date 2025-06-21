import gmsh
import numpy as np
from fealpy.mesh import TriangleMesh

def to_TriangleMesh():
    ntags, vxyz, _ = gmsh.model.mesh.getNodes()
    node = vxyz.reshape((-1,3))
    node = node[:,:2]
    vmap = dict({j:i for i,j in enumerate(ntags)})
    tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
    evid = np.array([vmap[j] for j in evtags])
    cell = evid.reshape((tris_tags.shape[-1],-1))
    return TriangleMesh(node,cell)
def triangle_domain():
    node = np.array([[0.0,0.0],[2.0,0.0],[1,np.sqrt(3)]],dtype=np.float64)
    cell = np.array([[0,1,2]],dtype=np.int_)
    mesh = TriangleMesh(node,cell)
    mesh.uniform_refine(2)
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    node[cell[-1,0]] = node[cell[-1,0]]+[-0.15,0.05]
    node[cell[-1,1]] = node[cell[-1,1]]+[-0.1,0.15]
    node[cell[-1,2]] = node[cell[-1,2]]+[0,-0.15]
    mesh.uniform_refine(3)
    return mesh
def unit_circle(h=0.1):
    gmsh.initialize()
    gmsh.model.occ.addDisk(0.0,0.0,0.0,1,1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
    gmsh.model.mesh.generate(2)

    mesh = to_TriangleMesh()
    gmsh.finalize()
    return mesh

def square_hole(h=0.05):
    gmsh.initialize()
    lc = h
    # 构建几何
    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
    gmsh.model.geo.addPoint(0, 1, 0, lc, 4)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(3, 2, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)

    gmsh.model.geo.addPoint(0.5,0.5,0,lc,5)
    gmsh.model.geo.addPoint(0.3,0.5,0,lc,6)
    gmsh.model.geo.addPoint(0.7,0.5,0,lc,7)

    gmsh.model.geo.addCircleArc(6,5,7,tag=5) # 生成圆弧
    gmsh.model.geo.addCircleArc(7,5,6,tag=6)# 该函数只能生成弧度小于等于180度的圆弧

    gmsh.model.geo.addCurveLoop([5,6],2)

    gmsh.model.geo.addPlaneSurface([1,2], 1)

    gmsh.model.geo.synchronize() 
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)

    gmsh.model.mesh.generate(2)
    mesh = to_TriangleMesh()
    gmsh.finalize()
    return mesh

def field_square_hole():
    gmsh.initialize()

    lc = 0.1

    gmsh.model.geo.addPoint(0.0,0.0,0.0,lc,1)
    gmsh.model.geo.addPoint(2.2,0.0,0.0,lc,2)
    gmsh.model.geo.addPoint(2.2,0.41,0.0,lc,3)
    gmsh.model.geo.addPoint(0.0,0.41,0.0,lc,4)

    gmsh.model.geo.addLine(1,2,1)
    gmsh.model.geo.addLine(2,3,2)
    gmsh.model.geo.addLine(3,4,3)
    gmsh.model.geo.addLine(4,1,4)

    gmsh.model.geo.addPoint(0.15,0.2,0.0,lc,5)
    gmsh.model.geo.addPoint(0.25,0.2,0.0,lc,6)
    gmsh.model.geo.addPoint(0.2,0.2,0.0,lc,7)

    gmsh.model.geo.add_circle_arc(6,7,5,5)
    gmsh.model.geo.add_circle_arc(5,7,6,6)

    gmsh.model.geo.addCurveLoop([1,2,3,4],1)
    gmsh.model.geo.add_curve_loop([5,6],2)

    gmsh.model.geo.addPlaneSurface([1,2],1)
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.field.add("Distance",1)
    gmsh.model.mesh.field.setNumbers(1,"CurvesList",[5,6])
    gmsh.model.mesh.field.setNumber(1,"Sampling",100)

    gmsh.model.mesh.field.add("Threshold",2)
    gmsh.model.mesh.field.setNumber(2,"InField",1)
    gmsh.model.mesh.field.setNumber(2,"SizeMin",0.0025)
    gmsh.model.mesh.field.setNumber(2,"SizeMax",0.01)
    gmsh.model.mesh.field.setNumber(2,"DistMin",0.01)
    gmsh.model.mesh.field.setNumber(2,"DistMax",0.02)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.model.mesh.generate(2)
    mesh = to_TriangleMesh()
    gmsh.finalize()
    return mesh


