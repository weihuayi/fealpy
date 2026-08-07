import argparse
import time

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

import AniMeshModel as MeshModel
from meshopt.TriAniMeshProblem import TriAniMeshProblem
from opt.PLBFGSAlg import PLBFGS
from opt.PNLCGAlg import PNLCG

import numpy as np # numpy is only used for data processing and visualization,not for core
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=
        '''
        Radius Ratio Optimization 2d Example
        ''')
parser.add_argument('--exam',
        default='tri', type=str,
        help=
        '''
        cir for circle domain,
        squ for square domain
        ''')

parser.add_argument('--optmethod',
        default= 'LBFGS', type=str,
        help='Optimization algorithm, default LBFGS, optional NLCG')

parser.add_argument('--mtype', 
            default=1, type=int,
            help='''Preprocessor type, 0 means do not use the preprocessor, 
                  1 means use the preprocessor''')

args = parser.parse_args()

def test_unit_circle(optmethod='LBFGS',mtype=1):
    mesh = MeshModel.unit_circle()

    def project_to_boundary(node):
        boundary_node_flag = mesh.boundary_node_flag()
        node[boundary_node_flag] = node[boundary_node_flag]/bm.linalg.norm(node[boundary_node_flag],axis=1,keepdims=True)
        return node
    def tangent_project1d(grad,node):
        boundary_node_flag = mesh.boundary_node_flag()
        egrad = grad[boundary_node_flag]
        r = node[boundary_node_flag]
        normal = r/bm.linalg.norm(r,axis=1,keepdims=True)
        gtd = bm.sum(egrad*normal,axis=1,keepdims=True)
        gradtan = egrad-gtd*normal
        grad[boundary_node_flag] = gradtan
        return grad

    def metric1(node):
        r = bm.sqrt(node[:,0]**2+node[:,1]**2)
        phi = 6*r
        m = bm.zeros((node.shape[0],2,2),dtype=bm.float64)

        m[:,0,0] = 25*bm.cos(phi)*bm.cos(phi)+bm.sin(phi)*bm.sin(phi)
        m[:,1,1] = 25*bm.sin(phi)*bm.sin(phi)+bm.cos(phi)*bm.cos(phi)
        m[:,0,1] = -24*bm.cos(phi)*bm.sin(phi)
        m[:,1,0] = -24*bm.cos(phi)*bm.sin(phi)
        return m

    def metric2(node):
        theta = bm.arctan2(node[:,1],node[:,0])
        phi = 6*theta
        m = bm.zeros((node.shape[0],2,2),dtype=bm.float64)

        m[:,0,0] = 25*bm.cos(phi)*bm.cos(phi)+bm.sin(phi)*bm.sin(phi)
        m[:,1,1] = 25*bm.sin(phi)*bm.sin(phi)+bm.cos(phi)*bm.cos(phi)
        m[:,0,1] = -24*bm.cos(phi)*bm.sin(phi)
        m[:,1,0] = -24*bm.cos(phi)*bm.sin(phi)
        return m

    isBdEdgeNode = mesh.boundary_node_flag()
    if mtype==1:
        metric0 = metric1
    if mtype==2:
        metric0 = metric2
    options = TriAniMeshProblem.get_options(
            mesh=mesh,
            Metric=metric0,
            Project=project_to_boundary,
            isBdEdgeNode=isBdEdgeNode,
            Tangent1d=tangent_project1d) 
    problem = TriAniMeshProblem(options)
    q = TriAniMeshProblem.get_quality(mesh,metric0)
    mesh.celldata["radius_ratio_init_quality"] = q
    mesh.to_vtk('init_ani_circle.vtu')
    show_mesh_quality(q,ylim=1750,title='Initial Mesh Quality',filename='init_ani_circle_quality')

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
        n = len(x)//2 
        node[:,0] = x[:n]
        node[:,1] = x[n:]
        problem.mesh = mesh
        if not problem.flipopt():
            break
        else:
            problem.x0 = node.T.flatten()
            mesh = problem.mesh
    t2 = time.time()
    print('optimize time:',t2-t1)
    node = mesh.entity('node')
    isFreeNode = ~mesh.boundary_node_flag()
    n = len(x)//2 
    node[:,0] = x[:n]
    node[:,1] = x[n:]

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()

    q = TriAniMeshProblem.get_quality(mesh,metric0)
    show_mesh_quality(q,ylim=1750,title='Radius Ratio Optimize Mesh Quality',filename='opt_ani_circle_quality')

    mesh.celldata["radius_ratio_opt_quality"] = q
    mesh.to_vtk(fname='optani_circle.vtu')  # Save the optimized mesh to a VTK
    return mesh

def test_unit_square(P=0,optmethod='LBFGS',mtype=1):
    mesh = MeshModel.unit_square(0.02)
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()
    isBdNode = mesh.boundary_node_flag()
    node = mesh.entity('node')
    isBdNode = mesh.boundary_node_flag()
 
    lst0 = bm.array([[1.0,0.0]])
    lst1 = bm.array([[0.0,1.0]])
    sqtag0 = bm.abs(node)<1e-8
    sqtag1 = bm.abs(node-1.0)<1e-8
    edgetag0 = sqtag0[:,1]|sqtag1[:,1]
    edgetag1 = sqtag0[:,0]|sqtag1[:,0]
    isFixNode = bm.sum(sqtag0|sqtag1,axis=1)==2
    edgetag0[isFixNode]=False
    edgetag1[isFixNode]=False
    isBdEdgeNode = edgetag0|edgetag1 
    def tangent_project1d(grad,node):
        egrad0 = grad[edgetag0]
        egrad1 = grad[edgetag1]
        egtd0 = bm.sum(egrad0*lst0,axis=1,keepdims=True)
        egtd1 = bm.sum(egrad1*lst1,axis=1,keepdims=True)
        gradtan0 = egtd0*lst0
        gradtan1 = egtd1*lst1
        grad[edgetag0] = gradtan0
        grad[edgetag1] = gradtan1
        return grad

    def project_to_boundary(node):
        node[sqtag0[:,1]&(~isFixNode),1] = 0.0
        node[sqtag1[:,1]&(~isFixNode),1] = 1.0
        node[sqtag0[:,0]&(~isFixNode),0] = 0.0
        node[sqtag1[:,0]&(~isFixNode),0] = 1.0
        return node

    def metric1(node):
        def f0(node):
            sigmay = 0.05
            sigmax = 0.3
            f = 1+(10-1)*bm.exp(-(node[:,1]-0.5)**2/(2*(sigmay*sigmay)))
            return f
        m = bm.zeros((node.shape[0],2,2),dtype=bm.float64)
        m0 = f0(node)
        m[:,0,0] = 3
        m[:,1,1] = m0
        return m

    def metric2(node):
        f = node[:,1]-0.5-0.15*bm.sin(2*bm.pi*node[:,0])
        gf = bm.zeros_like(node,dtype=bm.float64)
        gf[:,0] = -0.3*bm.pi*bm.cos(2*bm.pi*node[:,0])
        gf[:,1] = 1.0
        nf = gf/bm.linalg.norm(gf,axis=1,keepdims=True)
        t = bm.zeros_like(nf,dtype=bm.float64)
        t[:,0] = -nf[:,1]
        t[:,1] = nf[:,0]
        Q = bm.zeros((node.shape[0],2,2),dtype=bm.float64)
        Q[:,0,0] = t[:,0]
        Q[:,0,1] = t[:,1]
        Q[:,1,0] = nf[:,0]
        Q[:,1,1] = nf[:,1]
        T = bm.zeros((node.shape[0],2,2),dtype=bm.float64)
        T[:,0,0] = 1
        T[:,1,1] = 1+8*bm.exp(-f*f/(0.12*0.12))
        m = bm.einsum('nji,njk,nkl->nil', Q, T, Q)
        return m
    if mtype==1:
        metric0 = metric1
    elif mtype==2:
        metric0 = metric2

    options = TriAniMeshProblem.get_options(
                mesh=mesh,
                Metric = metric0,
                isFixNode = isFixNode,
                Project=project_to_boundary,
                isBdEdgeNode = isBdEdgeNode,
                Tangent1d = tangent_project1d)
    problem = TriAniMeshProblem(options)

    q = TriAniMeshProblem.get_quality(mesh,metric0)
    mesh.celldata['radius_ratio_init_quality'] = q
    mesh.to_vtk(fname='initani_squmesh.vtu')

    show_mesh_quality(q,ylim=3500,title='Initial Mesh Quality',filename='init_ani_squ_quality')
    NDof = len(problem.x0)
    problem.Preconditioner = None
    problem.Pi = None
    problem.StepLength = 1.0
    problem.FunValDiff = 1e-4
    problem.MaxIters = 10000
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
        n = len(x)//2 
        node[isFreeNode,0] = x[:n]
        node[isFreeNode,1] = x[n:]
        problem.mesh = mesh
        if not problem.flipopt():
            break
        else:
            problem.x0 = node[isFreeNode].T.flatten()
            mesh = problem.mesh

    t2 = time.time()
    print('optimize time:',t2-t1)

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()

    q = TriAniMeshProblem.get_quality(mesh,metric0)
    show_mesh_quality(q,ylim=3500,title='Radius Ratio Optimize Mesh Quality',filename='opt_ani_squ_quality')

    mesh.celldata["radius_ratio_opt_quality"] = q
    mesh.to_vtk(fname='optani_square.vtu')  # Save the optimized mesh to a VTK
    return mesh

def show_mesh_quality(q1,ylim=8000,title=None,filename=None):
    fig,axes= plt.subplots()
    q1 = bm.to_numpy(q1)
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
    return 0
if args.exam=='cir':
    test_unit_circle(optmethod=args.optmethod,mtype=args.mtype)
if args.exam=='squ':
    test_unit_square(optmethod=args.optmethod,mtype=args.mtype)

