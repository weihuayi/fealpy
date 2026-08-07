import argparse
import time

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

import MeshModel
from meshopt.TriMeshProblem import TriMeshProblem
from meshopt.TriSurfMeshProblem import TriSurfMeshProblem
from opt.PLBFGSAlg import PLBFGS
from opt.PNLCGAlg import PNLCG

import numpy as np # numpy is only used for data processing and visualization,not for core
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=
        '''
        Radius Ratio Optimization Example
        ''')
parser.add_argument('--exam',
        default='tri', type=str,
        help='''
        The default example is tri,
        tri for triangle domain, 
        sq for square hole, 
        ''')

parser.add_argument('--optmethod',
        default= 'LBFGS', type=str,
        help='Optimization algorithm, default LBFGS, optional NLCG')

parser.add_argument('--p', 
            default=0, type=int,
            help='''Preprocessor type, 0 means do not use the preprocessor, 
                  1 means use the preprocessor''')

args = parser.parse_args()

def test_square_hole(optmethod='LBFGS', P=0):
    mesh = MeshModel.square_hole(h=0.05)
    isBdNode = mesh.boundary_node_flag()
    node = mesh.entity('node')
    np.random.seed(0)
    node[~isBdNode] += 0.01*np.random.rand(node[~isBdNode].shape[0],node[~isBdNode].shape[1])
    mesh.uniform_refine(2)
    mesh.to_vtk(fname='sqh_init.vtu')
    node = mesh.entity('node')
    isBdNode = mesh.boundary_node_flag()
    q = TriMeshProblem.get_quality(mesh)
    show_mesh_quality(q,ylim=14000,title='Initial Mesh Quality',filename='square_hole_init')
 
    lst0 = np.array([[1.0,0.0]])
    lst1 = np.array([[0.0,1.0]])
    circlecenter = np.array([[0.5,0.5]])
    sqtag0 = np.abs(node)<1e-8
    sqtag1 = np.abs(node-1.0)<1e-8
    edgetag0 = sqtag0[:,1]|sqtag1[:,1]
    edgetag1 = sqtag0[:,0]|sqtag1[:,0]
    isFixNode = np.sum(sqtag0|sqtag1,axis=1)==2
    edgetag0[isFixNode]=False
    edgetag1[isFixNode]=False
    edgetag2 = isBdNode.copy()
    edgetag2[edgetag0] = False
    edgetag2[edgetag1] = False
    edgetag2[isFixNode] = False
    isBdEdgeNode = edgetag0|edgetag1|edgetag2 
    def tangent_project1d(grad,node):
        egrad0 = grad[edgetag0]
        egrad1 = grad[edgetag1]
        egrad2 = grad[edgetag2]
        egtd0 = np.sum(egrad0*lst0,axis=1,keepdims=True)
        egtd1 = np.sum(egrad1*lst1,axis=1,keepdims=True)
        gradtan0 = egtd0*lst0
        gradtan1 = egtd1*lst1
        grad[edgetag0] = gradtan0
        grad[edgetag1] = gradtan1
        r = node[edgetag2] - circlecenter
        normal = r/np.linalg.norm(r,axis=1,keepdims=True)
        gtd = np.sum(egrad2*normal,axis=1,keepdims=True)
        gradtan2 = egrad2 - gtd*normal
        grad[edgetag2] = gradtan2
        return grad

    def project_to_boundary(node):
        node[edgetag2] = circlecenter + 0.2*(node[edgetag2]-circlecenter)/np.linalg.norm(node[edgetag2]-circlecenter,axis=1,keepdims=True)
        node[sqtag0[:,1]&(~isFixNode),1] = 0.0
        node[sqtag1[:,1]&(~isFixNode),1] = 1.0
        node[sqtag0[:,0]&(~isFixNode),0] = 0.0
        node[sqtag1[:,0]&(~isFixNode),0] = 1.0
        return node

    options = TriMeshProblem.get_options(
            mesh = mesh,
            isFixNode = isFixNode,
            Project = project_to_boundary,
            isBdEdgeNode=isBdEdgeNode,
            Tangent1d=tangent_project1d)
    problem = TriMeshProblem(options)
    if P==1:
        problem.Preconditioner = problem.build_preconditioner(
                update_interval=3,
                rtol=1e-2,
                maxiter=100)
    elif P==0:
        problem.Preconditioner = None    
    problem.StepLength = 1.0
    problem.FunValDiff = 1e-6
    problem.MaxIters = 200
    problem.Print = False
    if optmethod == 'LBFGS':
        opt = PLBFGS(problem)
        opt.update_interval = 1
    if optmethod == 'NLCG':
        opt = PNLCG(problem)
    t1 = time.time()
    x,f,g = opt.run()
    t2 = time.time()
    print('optimize time:',t2-t1)
    node = mesh.entity('node')
    isFreeNode = ~isFixNode
    n = len(x)//2
    node[isFreeNode,0] = x[:n]
    node[isFreeNode,1] = x[n:]

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()

    mesh.to_vtk(fname='sqh_opt.vtu')
    q = TriMeshProblem.get_quality(mesh)
    show_mesh_quality(q,ylim=14000,title='Radius Ratio Optimize Mesh Quality',filename='square_hole_opt')
    return mesh

def test_triangle_domain(optmethod='LBFGS', P=0):
    mesh = MeshModel.triangle_domain()
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()
    q = TriMeshProblem.get_quality(mesh)
    show_mesh_quality(q,ylim=500,title='Radius Ratio Optimize Mesh Quality',filename='tri_init')

    mesh.to_vtk(fname='tri_domain_init.vtu')
    isFixNode = mesh.boundary_node_flag()
    options = TriMeshProblem.get_options(
            FixAllBoundary=True,
            mesh=mesh,
            isFixNode=isFixNode)
    problem = TriMeshProblem(options)
    if P==1:
        problem.Preconditioner = problem.build_preconditioner(
                update_interval=3,
                rtol=1e-2,
                maxiter=100)
    elif P==0:
        problem.Preconditioner = None    
    problem.StepLength = 1.0
    problem.FunValDiff = 1e-6
    problem.MaxIters = 200
    problem.Print = False
    if optmethod == 'LBFGS':
        opt = PLBFGS(problem)
    if optmethod == 'NLCG':
        opt = PNLCG(problem)
    t1 = time.time()
    x,f,g = opt.run()
    t2 = time.time()
    print('optimize time:',t2-t1)
    node = mesh.entity('node')
    isFreeNode = ~isFixNode
    n = len(x)//2
    node[isFreeNode,0] = x[:n]
    node[isFreeNode,1] = x[n:]
    
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.savefig('triangle_domain_rro.png', dpi=300, bbox_inches='tight')
    plt.close()
    q = TriMeshProblem.get_quality(mesh)
    show_mesh_quality(q,ylim=500,title='Radius Ratio Optimize Mesh Quality',filename='tri_opt')

    mesh.to_vtk(fname='tri_domain_opt.vtu')
    return mesh

def test_unit_sphere_surf(h=0.1, optmethod='LBFGS', P=0): 
    def project_to_boundary(node):
        node = node/bm.linalg.norm(node,axis=1,keepdims=True)
        return node
    def normal_vector2d(node):
        normal = bm.linalg.norm(node,axis=1,keepdims=True)
        normal[normal==0.0] = 1.0
        normals = node / normal
        return normals

    mesh = MeshModel.unit_sphere_surface(h)
    node = mesh.entity('node')
    NN = mesh.number_of_nodes()
    cell = mesh.entity('cell')
    isBdFaceNode = bm.ones(NN,dtype=bool)
    options = TriSurfMeshProblem.get_options(
            mesh = mesh,
            Project = project_to_boundary,
            isBdFaceNode = isBdFaceNode,
            Normal2d = normal_vector2d)
    problem = TriSurfMeshProblem(options)
    if P==1:
        problem.Preconditioner = problem.build_preconditioner(
                update_interval=1,
                rtol=1e-2,
                maxiter=100)
    elif P==0:
        problem.Preconditioner = None
    q = TriSurfMeshProblem.get_quality(mesh)
    show_mesh_quality(q,ylim=3000,title='Initial Mesh Quality',filename='init_sphere_surf_quality') 
    mesh.celldata['q'] = 1/q
    mesh.to_vtk(fname='init_sphere_surf.vtu') 

    problem.StepLength = 1.0
    problem.FunValDiff = 1e-6
    problem.MaxIters = 200
    problem.Print = True
    
    t1 = time.time()
    if optmethod == 'LBFGS':
        opt = PLBFGS(problem)
    if optmethod == 'NLCG':
        opt = PNLCG(problem)
    x, f, g = opt.run()
    t2 = time.time()
    print('optimize time:',t2-t1)
    node = mesh.entity('node')
    n = len(x)//3 
    node[:,0] = x[:n]
    node[:,1] = x[n:2*n]
    node[:,2] = x[2*n:]
    q = TriSurfMeshProblem.get_quality(mesh)
    mesh.celldata['quality'] = 1/q
    mesh.to_vtk(fname='opt_sphere_surf.vtu')
    show_mesh_quality(q,ylim=3000,title='Radius Ratio Mesh Optimize',filename='opt_sphere_surf_quality')
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
    filename = filename+f"p_{args.p}.png" 
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return 0

# main function to run the tests
if args.exam == 'tri':
    mesh = test_triangle_domain(optmethod=args.optmethod, P=args.p)
if args.exam == 'sq':
    mesh = test_square_hole(optmethod=args.optmethod, P=args.p)
if args.exam == 'sps':
    mesh = test_unit_sphere_surf(optmethod=args.optmethod, P=args.p)

