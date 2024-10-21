import numpy as np
import gmsh
import os
import sys
from fealpy.mesh import TetrahedronMesh

def to_TetrahedronMesh():
    ntags, vxyz, _ = gmsh.model.mesh.getNodes()
    node = vxyz.reshape((-1,3))
    vmap = dict({j:i for i,j in enumerate(ntags)})
    tets_tags,evtags = gmsh.model.mesh.getElementsByType(4)
    evid = np.array([vmap[j] for j in evtags])
    cell = evid.reshape((tets_tags.shape[-1],-1))
    return TetrahedronMesh(node,cell)

def unit_sphere(h=0.1,gopt:int=1):
    gmsh.initialize()
    gmsh.model.occ.addSphere(0.0,0.0,0.0,1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
    gmsh.option.setNumber("Mesh.Optimize",gopt)
    gmsh.model.mesh.generate(3)

    mesh = to_TetrahedronMesh()
    gmsh.finalize()
    return mesh
    
def LShape_ball(h=0.05,gopt:int=1):
    gmsh.initialize()
    gmsh.model.occ.addBox(0,0,0,1,1,1,1)
    gmsh.model.occ.addBox(0,0,0,0.5,0.5,0.5,2)
    gmsh.model.occ.addSphere(0.75,0.75,0.75,0.09,3)
    gmsh.model.occ.cut([(3,1)],[(3,2),(3,3)],4)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
    gmsh.option.setNumber("Mesh.Optimize",gopt)
    gmsh.model.mesh.generate(3)
    mesh = to_TetrahedronMesh()
    return mesh

def intersect_spheres(h=0.1,gopt:int=1):
    gmsh.initialize()
    gmsh.model.occ.addSphere(1.0,0.0,0.0,0.7,1)
    gmsh.model.occ.addSphere(-1.0,0.0,0.0,0.7,2)
    gmsh.model.occ.addSphere(0.5, 0.866025403784439,0.0,0.7,3)
    gmsh.model.occ.addSphere(-0.5,0.866025403784439,0.0,0.7,4)
    gmsh.model.occ.addSphere(0.5,-0.866025403784439,0.0,0.7,5)
    gmsh.model.occ.addSphere(-0.5,-0.866025403784439,0.0,0.7,6)
    gmsh.model.occ.addSphere(2.0, 0.0,0.0,0.7,7)
    gmsh.model.occ.addSphere(-2.0,0.0,0.0,0.7,8)
    gmsh.model.occ.addSphere(1.0,1.73205080756888,0.0,0.7,9)
    gmsh.model.occ.addSphere(-1.0,1.73205080756888,0.0,0.7,10)
    gmsh.model.occ.addSphere(-1.0,-1.73205080756888,0.0,0.7,11)
    gmsh.model.occ.addSphere(1.0,-1.73205080756888,0.0,0.7,12)
    gmsh.model.occ.fuse([(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7)],[(3,8),(3,9),(3,10),(3,11),(3,12)],13)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
    gmsh.option.setNumber("Mesh.Optimize",gopt)
    gmsh.model.mesh.generate(3)
    mesh = to_TetrahedronMesh()
    return mesh

def bunny(gopt:int=1):
    gmsh.initialize()

    # load step file
    path = os.path.dirname(os.path.abspath(__file__))
    gmsh.open(os.path.join(path, 'bunny.stp'))
    # uncomment this to fragment all volumes, i.e. make the geometry conformal
    # gmsh.model.occ.removeAllDuplicates()
    # gmsh.model.occ.synchronize()

    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 6)
    # 若 Mesh.MeshSizeFromCurvature 设为正值(默认为0),则网格将根据模型实体的
    # 曲率进行调整，该值给出每 2 Pi 弧度的目标元素数。
    gmsh.option.setNumber('Mesh.MeshSizeMax', 20)
    gmsh.option.setNumber("Mesh.Optimize",gopt)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 15)

    # get all model entities
    ent = gmsh.model.getEntities()

    physicals = {}
    for e in ent:
        n = gmsh.model.getEntityName(e[0], e[1])
        # get entity labels read from STEP and create a physical group for all
        # entities having the same 3rd label in the /-separated label path
        if n:
            print('Entity ' + str(e) + ' has label ' + n + ' (and mass ' +
                  str(gmsh.model.occ.getMass(e[0], e[1])) + ')')
            path = n.split('/')
            if e[0] == 3 and len(path) > 3:
                if (path[2] not in physicals):
                    physicals[path[2]] = []
                physicals[path[2]].append(e[1])

    # create the physical groups
    for name, tags in physicals.items():
        p = gmsh.model.addPhysicalGroup(3, tags)
        gmsh.model.setPhysicalName(3, p, name)

    gmsh.model.mesh.generate(3)
    gmsh.write("bunny.msh")
    #mesh = project()
    mesh = to_TetrahedronMesh()
    return mesh

def stlmodel():
    gmsh.initialize()
    path = os.path.dirname(os.path.abspath(__file__))
    gmsh.merge(os.path.join(path,'output.stl'))# 读取stl文件

    s = gmsh.model.getEntities(2) # 获得所有面实体
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s]) # 得到面环

    gmsh.model.geo.addVolume([l]) # 生成体实体
    gmsh.model.geo.synchronize() 

    gmsh.model.mesh.generate(3) # 生成网格
    mesh = to_TetrahedronMesh()
    return mesh
