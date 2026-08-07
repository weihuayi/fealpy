import argparse
import time

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh

import AniMeshModel as MeshModel
from meshopt.TetAniMeshProblem import TetAniMeshProblem
from opt.PLBFGSAlg import PLBFGS
from opt.PNLCGAlg import PNLCG

import numpy as np # numpy is only used for data processing and visualization,not for core
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=
        '''
        Radius Ratio Optimization Example
        ''')
parser.add_argument('--exam',
        default='sp', type=str,
        help='''
        The default example is a sphere,
        sp for sphere, cube for cube,
        ''')

parser.add_argument('--optmethod',
        default= 'LBFGS', type=str,
        help='Optimization algorithm, default LBFGS, optional NLCG')

parser.add_argument('--mtype',
        default= 1, type=int,
        help='Metric type, default 0, optional 1-2, see the code for details')

args = parser.parse_args()

def test_unit_sphere(h=0.1,optmethod='LBFGS', mtype=1):
    mesh = MeshModel.unit_sphere(h)
    
    def project_to_boundary(node):
        boundary_node_flag = mesh.boundary_node_flag()
        node[boundary_node_flag] = node[boundary_node_flag]/bm.linalg.norm(node[boundary_node_flag],axis=1,keepdims=True)
        return node
    def normal_vector(node):
        normal = bm.linalg.norm(node,axis=1,keepdims=True)
        normal[normal==0.0] = 1.0
        normals = node / normal
        return normals

    def metric1(node):
        r = node/(bm.linalg.norm(node,axis=1,keepdims=True)+1e-8)
        rr = bm.einsum('rk,rj->rkj',r,r) 
        lambda1 = 1
        lambda2 = 16
        M = lambda2*bm.eye(3) + (lambda1 - lambda2)*rr
        return M
    def metric2(node):
        d = 1-bm.linalg.norm(node,axis=1,keepdims=True)
        r = node/(bm.linalg.norm(node,axis=1,keepdims=True)+1e-8)
        rr = bm.einsum('rk,rj->rkj',r,r)
        lambdan = 1 + 20*bm.exp(-10*d)
        lambdat = 1
        M = lambdat*bm.eye(3) + (lambdan - lambdat)[:,None]*rr
        return M
    NC = mesh.number_of_cells()
    ylim = int(0.1*NC)
    if mtype==1:
        metric0 = metric1
    elif mtype==2:
        metric0 = metric2

    q =  TetAniMeshProblem.get_quality(mesh,metric0)
    mesh.celldata['radius_ratio_init_quality'] = 1/q
    bc = mesh.entity_barycenter('cell')
    mesh.celldata['bcz'] = bc[:,2]
    mesh.to_vtk(fname='init_sphere.vtu')

    show_mesh_quality(q,ylim=2500,title='Initial Mesh Quality',filename='init_sphere_quality')
    isBdFaceNode = mesh.boundary_node_flag()
    options = TetAniMeshProblem.get_options(
            mesh=mesh,
            Project = project_to_boundary,
            Normal2d = normal_vector,
            isBdFaceNode = isBdFaceNode,
            Metric = metric0)
    problem = TetAniMeshProblem(options)
    NDof = len(problem.x0)
    
    problem.Preconditioner =None
    problem.Pi = None
    problem.StepLength = 1.0
    problem.FunValDiff = 1e-4
    problem.MaxIters = 1000
    problem.Print = True
    t1 = time.time() 
    if optmethod == 'LBFGS':
        opt = PLBFGS(problem)
    if optmethod == 'NLCG':
        opt = PNLCG(problem)
    while True:
        x, f, g = opt.run()
        node = mesh.entity('node')
        n = len(x)//3
        node[:,0] = x[:n]
        node[:,1] = x[n:2*n]
        node[:,2] = x[2*n:]
        q =  TetAniMeshProblem.get_quality(mesh,metric0)
        problem.mesh = mesh
        if not problem.flipopt():
            break
        else:
            problem.x0 = node.T.flatten()
            mesh = problem.mesh
    t2 = time.time()
    print('optimize time:',t2-t1)
    node = mesh.entity('node')
    n = len(x)//3 
    node[:,0] = x[:n]
    node[:,1] = x[n:2*n]
    node[:,2] = x[2*n:]
    q =  TetAniMeshProblem.get_quality(mesh,metric0)
    mesh.celldata['radius_ratio_opt_quality'] = 1/q
    bc = mesh.entity_barycenter('cell')
    mesh.celldata['bcz'] = bc[:,2]
    mesh.to_vtk(fname='opt_sphere.vtu')
    
    show_mesh_quality(q,ylim=2500,title='Radius Ratio Mesh Optimize',filename='opt_sphere_quality')
    return mesh

def test_square3d(h=0.05, optmethod='LBFGS', mtype=1):
    mesh = MeshModel.unit_square3d(h)
    node = mesh.entity('node')
    nodeinit=node.copy()
    isBdNode = mesh.boundary_node_flag()
    lst0 = bm.array([[1.0,0.0,0.0]])
    lst1 = bm.array([[0.0,1.0,0.0]])
    lst2 = bm.array([[0.0,0.0,1.0]])
    sqtag0 = bm.abs(node)<1e-8
    sqtag1 = bm.abs(node-1.0)<1e-8
    edgetag0 = (sqtag0[:,1]&sqtag0[:,2])|(sqtag0[:,1]&sqtag1[:,2])\
            |(sqtag1[:,1]&sqtag0[:,2])|(sqtag1[:,1]&sqtag1[:,2])
    edgetag1 = (sqtag0[:,0]&sqtag0[:,2])|(sqtag1[:,0]&sqtag0[:,2])\
            |(sqtag0[:,0]&sqtag1[:,2])|(sqtag1[:,0]&sqtag1[:,2])
    edgetag2 = (sqtag0[:,0]&sqtag0[:,1])|(sqtag1[:,0]&sqtag0[:,1])\
            |(sqtag0[:,0]&sqtag1[:,1])|(sqtag1[:,0]&sqtag1[:,1])
    isFixNode = bm.sum(sqtag0|sqtag1,axis=1)==3
    edgetag0[isFixNode] = False
    edgetag1[isFixNode] = False
    edgetag2[isFixNode] = False
    isBdEdgeNode = edgetag0|edgetag1|edgetag2
    isBdFaceNode = isBdNode & ~(isBdEdgeNode|isFixNode)
    def tangent_project1d(grad,node):
        egrad0 = grad[edgetag0]
        egrad1 = grad[edgetag1]
        egrad2 = grad[edgetag2]
        egtd0 = bm.sum(egrad0*lst0,axis=1,keepdims=True)
        egtd1 = bm.sum(egrad1*lst1,axis=1,keepdims=True)
        egtd2 = bm.sum(egrad2*lst2,axis=1,keepdims=True)
        gradtan0 = egtd0*lst0
        gradtan1 = egtd1*lst1
        gradtan2 = egtd2*lst2

        grad[edgetag0] = gradtan0
        grad[edgetag1] = gradtan1
        grad[edgetag2] = gradtan2
        return grad

    def project_to_boundary(node):
        node[isFixNode] = nodeinit[isFixNode]
        if bm.any(edgetag0):
            y0 = nodeinit[edgetag0,1]
            z0 = nodeinit[edgetag0,2]
            node[edgetag0,1] = y0
            node[edgetag0,2] = z0
        if bm.any(edgetag1):
            x1 = nodeinit[edgetag1,0]
            z1 = nodeinit[edgetag1,2]
            node[edgetag1,0] = x1
            node[edgetag1,2] = z1
        if bm.any(edgetag2):
            x2 = nodeinit[edgetag2,0]
            y2 = nodeinit[edgetag2,1]
            node[edgetag2,0] = x2
            node[edgetag2,1] = y2
        node[isBdFaceNode & sqtag0[:,0],0] = 0.0
        node[isBdFaceNode & sqtag1[:,0],0] = 1.0
        node[isBdFaceNode & sqtag0[:,1],1] = 0.0
        node[isBdFaceNode & sqtag1[:,1],1] = 1.0
        node[isBdFaceNode & sqtag0[:,2],2] = 0.0
        node[isBdFaceNode & sqtag1[:,2],2] = 1.0
        return node

    def metric1(node):
        M = bm.zeros((node.shape[0],3,3),dtype=bm.float64)
        Minv = bm.zeros((node.shape[0],3,3),dtype=bm.float64)
        def f0(node):
            sigmay = 0.05
            sigmax = 0.3
            f = 1+(10-1)*bm.exp(-(node[:,2]-0.5)**2/(2*(sigmay*sigmay)))
            return f
        def f1(node):
            sigmay = 0.05
            sigmax = 0.3
            f = 1+(10-1)*bm.exp(-(node[:,1]-0.5)**2/(2*(sigmay*sigmay)))
            return f
        M[:,0,0] = 3 
        M[:,1,1] = f1(node)
        M[:,2,2] = f0(node)
        return M

    if mtype==1:
        metric0 = metric1

    q =  TetAniMeshProblem.get_quality(mesh,metric0)
    mesh.celldata['radius_ratio_init_quality'] = 1/q
    mesh.to_vtk(fname='initial_squ3d.vtu')
    show_mesh_quality(q,ylim=3500,title='Initial Mesh Quality',filename='initial_squ3d_quality') 
    options = TetAniMeshProblem.get_options( 
            mesh=mesh,
            Metric = metric0,
            isFixNode = isFixNode,
            isBdEdgeNode = isBdEdgeNode,
            isBdFaceNode = isBdFaceNode,
            Project = project_to_boundary,
            Tangent1d = tangent_project1d)
    problem = TetAniMeshProblem(options)

    NDof = len(problem.x0)
    
    problem.Preconditioner = None
    problem.Pi = None
    problem.StepLength = 1.0
    problem.FunValDiff = 1e-4
    problem.MaxIters = 1000
    problem.Print = True
    
    t1 = time.time()
    if optmethod == 'LBFGS':
        opt = PLBFGS(problem)
    if optmethod == 'NLCG':
        opt = PNLCG(problem)
    while True:
        x, f, g = opt.run()
        node = mesh.entity('node')
        isFreeNode = ~isFixNode
        n = len(x)//3
        node[isFreeNode,0] = x[:n]
        node[isFreeNode,1] = x[n:2*n]
        node[isFreeNode,2] = x[2*n:]
        problem.mesh = mesh
        if not problem.flipopt():
            break
        else:
            problem.x0 = node[isFreeNode].T.flatten()
            mesh = problem.mesh
    t2 = time.time()
    print('optimize time:',t2-t1)
    node = mesh.entity('node')
    isFreeNode = ~isFixNode
    n = len(x)//3 
    node[isFreeNode,0] = x[:n]
    node[isFreeNode,1] = x[n:2*n]
    node[isFreeNode,2] = x[2*n:]

    q =  TetAniMeshProblem.get_quality(mesh,metric0)
    mesh.celldata['radius_ratio_opt_quality'] = 1/q
    mesh.to_vtk(fname='opt_squ3d.vtu')

    show_mesh_quality(q,ylim=3500,title='Radius Ratio Mesh Optimize',filename='opt_squ3d_quality')
    return mesh

def show_mesh_quality(q1,ylim=8000,title=None,filename=None):
    fig,axes= plt.subplots()
    q1 = bm.to_numpy(q1)
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
        axes.set_title(title, fontsize=16, pad=20)

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
    filename = filename 
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return 0# main function to run the tests

if args.exam == 'sp':
    mesh = test_unit_sphere(optmethod=args.optmethod,mtype=args.mtype)
if args.exam == 'cube':
    mesh = test_square3d(optmethod=args.optmethod,mtype=args.mtype)
