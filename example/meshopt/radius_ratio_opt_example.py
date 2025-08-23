import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import gmsh

from fealpy.backend import backend_manager as bm
from fealpy.opt import PLBFGS, PNLCG
from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.meshopt import RadiusRatioQuality
from fealpy.meshopt import RadiusRatioSumObjective
from fealpy.operator import LinearOperator

parser = argparse.ArgumentParser(description=
        '''
        半径比优化示例
        ''')
parser.add_argument('--exam',
            default='sp',type=str,
            help='''
            sp: 单位球示例
            tsp: 12个球相交示例
                ''')
parser.add_argument('--optmethod',
            default='LBFGS',type=str,
            help='''
            使用的优化算法
            LBFGS: 使用LBFGS算法
            NLCG: 使用NLCG算法
                ''')

parser.add_argument('--p', 
            default=1,type=int,
            help='预处理器类型，0表示不使用预处理器，1表示使用预处理器')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='默认后端为numpy')

parser.add_argument('--output',
        default=False, type=bool,
        help='是否输出.vtu可视化文件')

args = parser.parse_args()
bm.set_backend(args.backend)

def test_triangle_domain(P=0,optmethod='LBFGS'):
    mesh = MeshModel.triangle_domain()
    
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()

    mesh_quality = RadiusRatioQuality(mesh)
    sum_q = RadiusRatioSumObjective(mesh_quality)
    q = mesh_quality(mesh.node)
    NC = mesh.number_of_cells()
    
    show_mesh_quality(bm.to_numpy(q),ylim=2500,title='initial mesh quality') 

    NDof = len(sum_q.x0)
    if P==1:
        Preconditioner = LinearOperator((NDof,NDof),sum_q.preconditioner)
        update_Preconditioner = sum_q.update_preconditioner
    elif P==0:
        Preconditioner = None
        update_Preconditioner = None
    x0 = sum_q.x0
    if optmethod=='LBFGS':
        options = PLBFGS.get_options(
                x0=sum_q.x0,
                objective = sum_q.fun_with_grad,
                Preconditioner = Preconditioner, 
                update_Preconditioner = update_Preconditioner,
                MaxIters=100,
                FunValDiff = 1e-4)
        opt = PLBFGS(options)

    if optmethod=='NLCG':
        options = PNLCG.get_options(
                x0=sum_q.x0,
                objective = sum_q.fun_with_grad,
                Preconditioner = Preconditioner, 
                update_Preconditioner = update_Preconditioner,
                MaxIters=100,
                FunValDiff = 1e-4)
        opt = PLBFGS(options)

    t1 = time.time()
    x, f, g, flag = opt.run()
    t2 = time.time()
    print('time:',t2-t1)
    print('NF:',opt.NF)
    node = mesh.entity('node')
    isFreeNode = ~mesh.boundary_node_flag()
    n = len(x)//2 
    
    node = bm.set_at(node, (isFreeNode,0), x[:n])
    node = bm.set_at(node, (isFreeNode,1), x[n:])

    mesh_quality = RadiusRatioQuality(mesh)
    q = mesh_quality(node) 

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()

    show_mesh_quality(bm.to_numpy(q),ylim=2500,title='optimize mesh quality')
    return mesh

def test_square_hole(P=0, optmethod='LBFGS'):
    mesh = MeshModel.square_hole()
    
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()

    mesh_quality = RadiusRatioQuality(mesh)
    sum_q = RadiusRatioSumObjective(mesh_quality)
    q = mesh_quality(mesh.node)
    NC = mesh.number_of_cells()
    
    show_mesh_quality(bm.to_numpy(q),ylim=14000,title='initial mesh quality') 

    NDof = len(sum_q.x0)
    if P==1:
        Preconditioner = LinearOperator((NDof,NDof),sum_q.preconditioner)
        update_Preconditioner = sum_q.update_preconditioner
    elif P==0:
        Preconditioner = None
        update_Preconditioner = None
    x0 = sum_q.x0
    if optmethod=='LBFGS':
        options = PLBFGS.get_options(
                x0=sum_q.x0,
                objective = sum_q.fun_with_grad,
                Preconditioner = Preconditioner, 
                update_Preconditioner = update_Preconditioner,
                MaxIters=100,
                FunValDiff = 1e-4)
        opt = PLBFGS(options)

    if optmethod=='NLCG':
        options = PNLCG.get_options(
                x0=sum_q.x0,
                objective = sum_q.fun_with_grad,
                Preconditioner = Preconditioner, 
                update_Preconditioner = update_Preconditioner,
                MaxIters=100,
                FunValDiff = 1e-4)
        opt = PLBFGS(options)

    t1 = time.time()
    x, f, g, flag = opt.run()
    t2 = time.time()
    print('time:',t2-t1)
    print('NF:',opt.NF)
    node = mesh.entity('node')
    isFreeNode = ~mesh.boundary_node_flag()
    n = len(x)//2 
    
    node = bm.set_at(node, (isFreeNode,0), x[:n])
    node = bm.set_at(node, (isFreeNode,1), x[n:])

    mesh_quality = RadiusRatioQuality(mesh)
    q = mesh_quality(node) 

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()

    show_mesh_quality(bm.to_numpy(q),ylim=14000,title='optimize mesh quality')
    return mesh

def test_sphere(P=0,optmethod='LBFGS'):
    mesh = MeshModel.unit_sphere()
    
    mesh_quality = RadiusRatioQuality(mesh)
    sum_q = RadiusRatioSumObjective(mesh_quality)
    q = mesh_quality(mesh.node)
    NC = mesh.number_of_cells()
    
    show_mesh_quality(bm.to_numpy(q),ylim=2500,title='initial mesh quality') 

    NDof = len(sum_q.x0)
    if P==1:
        Preconditioner = LinearOperator((NDof,NDof),sum_q.preconditioner)
        update_Preconditioner = sum_q.update_preconditioner
    elif P==0:
        Preconditioner = None
        update_Preconditioner = None
    x0 = sum_q.x0
    if optmethod=='LBFGS':
        options = PLBFGS.get_options(
                x0=sum_q.x0,
                objective = sum_q.fun_with_grad,
                Preconditioner = Preconditioner, 
                update_Preconditioner = update_Preconditioner,
                MaxIters=100,
                FunValDiff = 1e-8)
        opt = PLBFGS(options)

    if optmethod=='NLCG':
        options = PNLCG.get_options(
                x0=sum_q.x0,
                objective = sum_q.fun_with_grad,
                Preconditioner = Preconditioner, 
                update_Preconditioner = update_Preconditioner,
                MaxIters=100,
                FunValDiff = 1e-8)
        opt = PLBFGS(options)
    t1 = time.time()
    x, f, g, flag = opt.run()
    t2 = time.time()
    print('time:',t2-t1)
    print('NF:',opt.NF)
    node = mesh.entity('node')
    isFreeNode = ~mesh.boundary_node_flag()
    n = len(x)//3 
    
    node = bm.set_at(node, (isFreeNode,0), x[:n])
    node = bm.set_at(node, (isFreeNode,1), x[n:2*n])
    node = bm.set_at(node, (isFreeNode,2), x[2*n:])

    mesh_quality = RadiusRatioQuality(mesh)
    q = mesh_quality(node) 

    show_mesh_quality(bm.to_numpy(q),ylim=2500,title='optimize mesh quality')
    return mesh


def test_sphere_intersection(P=0,optmethod='LBFGS'):
    mesh = MeshModel.sphere_intersection()
    
    mesh_quality = RadiusRatioQuality(mesh)
    sum_q = RadiusRatioSumObjective(mesh_quality)
    q = mesh_quality(mesh.node)
    NC = mesh.number_of_cells()
    
    show_mesh_quality(bm.to_numpy(q),ylim=9000,title='initial mesh quality') 

    NDof = len(sum_q.x0)
    if P==1:
        Preconditioner = LinearOperator((NDof,NDof),sum_q.preconditioner)
        update_Preconditioner = sum_q.update_preconditioner
    elif P==0:
        Preconditioner = None
        update_Preconditioner = None
    x0 = sum_q.x0
    if optmethod=='LBFGS':
        options = PLBFGS.get_options(
                x0=sum_q.x0,
                objective = sum_q.fun_with_grad,
                Preconditioner = Preconditioner, 
                update_Preconditioner = update_Preconditioner,
                MaxIters=100,
                FunValDiff = 1e-4)
        opt = PLBFGS(options)

    if optmethod=='NLCG':
        options = PNLCG.get_options(
                x0=sum_q.x0,
                objective = sum_q.fun_with_grad,
                Preconditioner = Preconditioner, 
                update_Preconditioner = update_Preconditioner,
                MaxIters=100,
                FunValDiff = 1e-4)
        opt = PLBFGS(options)
    t1 = time.time()
    x, f, g, flag = opt.run()
    t2 = time.time()
    print('time:',t2-t1)
    print('NF:',opt.NF)
    node = mesh.entity('node')
    isFreeNode = ~mesh.boundary_node_flag()
    n = len(x)//3 
    
    node = bm.set_at(node, (isFreeNode,0), x[:n])
    node = bm.set_at(node, (isFreeNode,1), x[n:2*n])
    node = bm.set_at(node, (isFreeNode,2), x[2*n:])

    mesh_quality = RadiusRatioQuality(mesh)
    q = mesh_quality(node) 

    show_mesh_quality(bm.to_numpy(q),ylim=9000,title='optimize mesh quality')
    return mesh

def show_mesh_quality(q1,ylim=8000,title=None):
    fig,axes= plt.subplots()
    q1 = 1/q1
    minq1 = np.min(q1)
    maxq1 = np.max(q1)
    meanq1 = np.mean(q1)
    rmsq1 = np.sqrt(np.mean(q1**2))
    stdq1 = np.std(q1)
    NC = len(q1)
    SNC = np.sum((q1<0.3))
    hist, bins = np.histogram(q1, bins=50, range=(0, 1))
    center = (bins[:-1] + bins[1:]) / 2
    axes.bar(center, hist, align='center', width=0.02)
    axes.set_xlim(0, 1)
    axes.set_ylim(0,ylim)

    if title is not None:
        axes.set_title(title,fontsize=16,pad=20)

    #TODO: fix the textcoords warning
    axes.annotate('Min quality: {:.6}'.format(minq1), xy=(0, 0),
            xytext=(0.15, 0.85),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    axes.annotate('Max quality: {:.6}'.format(maxq1), xy=(0, 0),
            xytext=(0.15, 0.8),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    axes.annotate('Average quality: {:.6}'.format(meanq1), xy=(0, 0),
            xytext=(0.15, 0.75),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    axes.annotate('RMS: {:.6}'.format(rmsq1), xy=(0, 0),
            xytext=(0.15, 0.7),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    axes.annotate('STD: {:.6}'.format(stdq1), xy=(0, 0),
            xytext=(0.15, 0.65),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    axes.annotate('radius radio less than 0.3:{:.0f}/{:.0f}'.format(SNC,NC), xy=(0, 0),
            xytext=(0.15, 0.6),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    plt.tight_layout()
    plt.show()
    return 0

class MeshModel():
    def __init__(self):
        pass
    @staticmethod
    def to_TetrahedronMesh():
        ntags, vxyz, _ = gmsh.model.mesh.getNodes()
        node = bm.tensor(vxyz.reshape((-1,3)))
        vmap = dict({j:i for i,j in enumerate(ntags)})
        tets_tags,evtags = gmsh.model.mesh.getElementsByType(4)
        evid = bm.tensor([vmap[j] for j in evtags])
        cell = bm.tensor(evid.reshape((tets_tags.shape[-1],-1))) 
        return TetrahedronMesh(node,cell)

    @classmethod
    def triangle_domain(cls):
        node = bm.array([[0.0,0.0],[2.0,0.0],[1,np.sqrt(3)]],dtype=bm.float64)
        cell = bm.array([[0,1,2]],dtype=bm.int32)
        mesh = TriangleMesh(node,cell)
        mesh.uniform_refine(2)
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        #node[cell[-1,0]] = node[cell[-1,0]]+[-0.15,0.05]
        #node[cell[-1,1]] = node[cell[-1,1]]+[-0.1,0.15]
        #node[cell[-1,2]] = node[cell[-1,2]]+[0,-0.15]
        node = bm.set_at(node, (cell[-1,0],slice(None)), node[cell[-1,0],:] +
                         bm.tensor([-0.15,0.05]))
        node = bm.set_at(node, (cell[-1,1],slice(None)), node[cell[-1,1],:] +
                         bm.tensor([-0.1,0.15]))
        node = bm.set_at(node, (cell[-1,2],slice(None)), node[cell[-1,2],:] +
                         bm.tensor([0,-0.15]))
        if args.backend=='jax':
            mesh = TriangleMesh(node,cell)
        mesh.uniform_refine(3)
        return mesh

    @classmethod
    def square_hole(cls):
        gmsh.initialize()
        lc = 0.05

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

        gmsh.model.geo.addCircleArc(6,5,7,tag=5)
        gmsh.model.geo.addCircleArc(7,5,6,tag=6)

        gmsh.model.geo.addCurveLoop([5,6],2)

        gmsh.model.geo.addPlaneSurface([1,2], 1)

        gmsh.model.geo.synchronize() 
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0),0.05)

        gmsh.model.mesh.generate(2)
        ntags, vxyz, _ = gmsh.model.mesh.getNodes()
        node = vxyz.reshape((-1,3))
        node = node[:,:2]
        node = np.delete(node,4,0)
        node = bm.tensor(node)
        vmap = dict({j:i for i,j in enumerate(ntags)})
        tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
        evid = bm.tensor([vmap[j] for j in evtags])
        cell = bm.tensor(evid.reshape((tris_tags.shape[-1],-1)))
        #cell[cell>4] = cell[cell>4]-1
        cell = bm.set_at(cell, (cell>4,), cell[cell>4]-1)
        mesh = TriangleMesh(node,cell)
        gmsh.finalize()
        isBdNode = mesh.boundary_node_flag()
        node = mesh.entity('node')
        np.random.seed(0)
        node = bm.set_at(node, (~isBdNode,slice(None)), node[~isBdNode,:] +
                         bm.tensor(0.01*np.random.rand(node[~isBdNode].shape[0],node[~isBdNode].shape[1])))
        if args.backend=='jax':
            mesh = TriangleMesh(node,cell)
        mesh.uniform_refine(2)
        return mesh

    @classmethod
    def unit_sphere(cls):
        gmsh.initialize()
        gmsh.model.occ.addSphere(0.0,0.0,0.0,1,1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0),0.1)
        gmsh.option.setNumber("Mesh.Optimize",0)
        gmsh.model.mesh.generate(3)

        mesh = cls.to_TetrahedronMesh()
        gmsh.finalize()
        return mesh

    @classmethod
    def sphere_intersection(cls):
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
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0),0.1)
        gmsh.option.setNumber("Mesh.Optimize",0)
        gmsh.model.mesh.generate(3)
        mesh = cls.to_TetrahedronMesh()
        gmsh.finalize()
        return mesh

if args.exam=='tri':
    mesh = test_triangle_domain(P=args.p,optmethod=args.optmethod)
if args.exam=='sq':
    mesh = test_square_hole(P=args.p,optmethod=args.optmethod)
if args.exam=='sp':
    mesh = test_sphere(P=args.p,optmethod=args.optmethod)
if args.exam=='tsp':
    mesh = test_sphere_intersection(P=args.p,optmethod=args.optmethod)
if args.output:
    mesh.to_vtk(fname='optmesh.vtu')
