import argparse
import time

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh

import MeshModel
from meshopt.TetMeshProblem import TetMeshProblem
from meshopt.TriMeshProblem import TriMeshProblem
from opt.PLBFGSAlg import PLBFGS
from opt.PNLCGAlg import PNLCG

import numpy as np # numpy is only used for data processing and visualization,not for core computations
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=
        '''
        Radius Ratio Optimization Example
        ''')
parser.add_argument('--exam',
        default='sp', type=str,
        help='''
        The default example is a sphere,
        sp for sphere, ls for L-shaped domain, 
        tsp for intersecting spheres,
        ''')


parser.add_argument('--optmethod',
        default= 'LBFGS', type=str,
        help='Optimization algorithm, default LBFGS, optional NLCG')

parser.add_argument('--p', 
            default=0, type=int,
            help='''Preprocessor type, 0 means do not use the preprocessor, 
                  1 means use the preprocessor''')

args = parser.parse_args()

def test_unit_sphere(h=0.1, optmethod='LBFGS', P=0):
    #filename = 'data/sphere_init.vtu'
    #mesh = TetrahedronMesh.from_vtu(filename)
    mesh = MeshModel.unit_sphere(h)
    def project_to_boundary(node):
        boundary_node_flag = mesh.boundary_node_flag()
        node[boundary_node_flag] = node[boundary_node_flag]/bm.linalg.norm(node[boundary_node_flag],axis=1,keepdims=True)
        return node
    def unit_normal2d(node):
        normal = bm.linalg.norm(node,axis=1,keepdims=True)
        normal[normal==0.0] = 1.0
        normals = node / normal
        return normals

    q = TetMeshProblem.get_quality(mesh)
    NC = mesh.number_of_cells()
    
    q = mesh.cell_quality()
    angle = mesh.dihedral_angle()
    min_angle = bm.min(angle,axis=1)
    show_angle(min_angle,ylim=3000,title='Initial Mesh Dihedral Angle',filename='sphere_init_angle')
    show_mesh_quality(q,ylim=3000,title='Initial Mesh Quality',filename='sphere_init') 

    isBdFaceNode = mesh.boundary_node_flag()
    isFixNode = bm.zeros(len(isBdFaceNode),dtype=bm.bool)
    options = TetMeshProblem.get_options(
            mesh=mesh,
            isFixNode = isFixNode,
            Project = project_to_boundary,
            isBdFaceNode = isBdFaceNode,
            Normal2d=unit_normal2d)
    problem = TetMeshProblem(options)
     
    NDof = len(problem.x0)
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
    problem.Print = True
    
    t1 = time.time()
    if optmethod == 'LBFGS':
        opt = PLBFGS(problem)
    if optmethod == 'NLCG':
        opt = PNLCG(problem)
    nflip=0
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
            print('翻转次数:',nflip)
            break
        else:
            nflip +=1
            problem.x0 = node.T.flatten()
            mesh = problem.mesh
            if P==1:
                problem.Preconditioner.reset()
    t2 = time.time()
    print('optimize time:',t2-t1)
    node = mesh.entity('node')
    isFreeNode = ~isFixNode
    n = len(x)//3 
    node[isFreeNode,0] = x[:n]
    node[isFreeNode,1] = x[n:2*n]
    node[isFreeNode,2] = x[2*n:]
    q = TetMeshProblem.get_quality(mesh)
    mesh.celldata['radius_ratio'] = 1/q
    mesh.to_vtk(fname='rro_sphere.vtu')
    angle = mesh.dihedral_angle()
    min_angle = bm.min(angle,axis=1)

    show_angle(min_angle,ylim=3000,title=' Radius Ratio Optimize Mesh Dihedral Angle',filename='sphere_opt_angle')
    show_mesh_quality(q,ylim=3000,title='Radius Ratio Optimize Mesh Quality',filename='sphere_opt')
    return mesh

def test_LShape(h=0.05, optmethod='LBFGS', P=0):
    #filename = 'data/lshape_init.vtu' 
    #mesh = TetrahedronMesh.from_vtu(filename)
    mesh = MeshModel.LShape(h)
    node = mesh.entity('node')
    nodeinit = node.copy()
    isBdNode = mesh.boundary_node_flag()
    lst0 = bm.array([[1.0,0.0,0.0]])
    lst1 = bm.array([[0.0,1.0,0.0]])
    lst2 = bm.array([[0.0,0.0,1.0]])
    tag0 = bm.abs(node)<1e-8
    tag1 = bm.abs(node-1.0)<1e-8
    tag2 = bm.abs(node-0.5)<1e-8
    lstag0 = bm.abs(node)<1e-8
    lstag1 = bm.abs(node-1.0)<1e-8
    lstag2 = bm.abs(node-0.5)<1e-8
   
    edgetag0 =(tag0[:,1]&tag0[:,2])|(tag2[:,1]&tag0[:,2])|(tag0[:,1]&tag2[:,2])\
    |(tag2[:,1]&tag2[:,2])|(tag1[:,1]&tag0[:,2])|(tag0[:,1]&tag1[:,2])|(tag1[:,1]&tag1[:,2])
    edgetag1 =(tag0[:,0]&tag0[:,2])|(tag2[:,0]&tag0[:,2])|(tag0[:,0]&tag2[:,2])\
    |(tag2[:,0]&tag2[:,2])|(tag1[:,0]&tag0[:,2])|(tag0[:,0]&tag1[:,2])|(tag1[:,0]&tag1[:,2])
    edgetag2 =(tag0[:,0]&tag0[:,1])|(tag2[:,0]&tag0[:,1])|(tag0[:,0]&tag2[:,1])\
    |(tag2[:,0]&tag2[:,1])|(tag1[:,0]&tag0[:,1])|(tag0[:,0]&tag1[:,1])|(tag1[:,0]&tag1[:,1])
     
    isFixNode = bm.sum(tag0|tag1|tag2,axis=1)==3
    edgetag0[isFixNode] = False
    edgetag1[isFixNode] = False
    edgetag2[isFixNode] = False
    isBdEdgeNode = edgetag0|edgetag1|edgetag2
    isBdFaceNode = isBdNode & ~(isFixNode|isBdEdgeNode)
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
        # ---------- 角点：直接固定 ----------
        node[isFixNode] = nodeinit[isFixNode]

        # ---------- 边点回投 ----------
        # edgetag0: 边方向平行 x 轴，所以 y,z 固定
        # 这些边满足 y,z 分别取 0 / 0.5 / 1 中某两个常值
        if bm.any(edgetag0):
            y0 = nodeinit[edgetag0, 1]
            z0 = nodeinit[edgetag0, 2]
            node[edgetag0, 1] = y0
            node[edgetag0, 2] = z0

        # edgetag1: 边方向平行 y 轴，所以 x,z 固定
        if bm.any(edgetag1):
            x0 = nodeinit[edgetag1, 0]
            z0 = nodeinit[edgetag1, 2]
            node[edgetag1, 0] = x0
            node[edgetag1, 2] = z0

        # edgetag2: 边方向平行 z 轴，所以 x,y 固定
        if bm.any(edgetag2):
            x0 = nodeinit[edgetag2, 0]
            y0 = nodeinit[edgetag2, 1]
            node[edgetag2, 0] = x0
            node[edgetag2, 1] = y0

        # ---------- 面点回投 ----------
        # 面点不是边点也不是角点
        face_x0  = isBdFaceNode & lstag0[:, 0]   # x = 0
        face_x05 = isBdFaceNode & lstag2[:, 0]   # x = 0.5
        face_x1  = isBdFaceNode & lstag1[:, 0]   # x = 1

        face_y0  = isBdFaceNode & lstag0[:, 1]   # y = 0
        face_y05 = isBdFaceNode & lstag2[:, 1]   # y = 0.5
        face_y1  = isBdFaceNode & lstag1[:, 1]   # y = 1

        face_z0  = isBdFaceNode & lstag0[:, 2]   # z = 0
        face_z05 = isBdFaceNode & lstag2[:, 2]   # z = 0.5
        face_z1  = isBdFaceNode & lstag1[:, 2]   # z = 1

        node[face_x0,  0] = 0.0
        node[face_x05, 0] = 0.5
        node[face_x1,  0] = 1.0

        node[face_y0,  1] = 0.0
        node[face_y05, 1] = 0.5
        node[face_y1,  1] = 1.0

        node[face_z0,  2] = 0.0
        node[face_z05, 2] = 0.5
        node[face_z1,  2] = 1.0

        return node
    
    NC = mesh.number_of_cells()
    q = TetMeshProblem.get_quality(mesh)
    angle = mesh.dihedral_angle()
    min_angle = bm.min(angle,axis=1)
    show_angle(min_angle,ylim=5000,title='Initial Mesh Dihedral Angle',filename='lshape_init_angle')

    show_mesh_quality(q,ylim=5000,title='Initial Mesh Quality',filename='lshape_init') 

    options = TetMeshProblem.get_options(
            mesh=mesh,
            isFixNode = isFixNode,
            isBdEdgeNode = isBdEdgeNode,
            isBdFaceNode = isBdFaceNode,
            Project = project_to_boundary,
            Tangent1d=tangent_project1d)
    problem = TetMeshProblem(options)

    NDof = len(problem.x0)
    if P==1:
        problem.Preconditioner = problem.build_preconditioner(
                update_interval=3,
                rtol=1e-2,
                maxiter=100)
    elif P==0:
        problem.Preconditioner = None
    problem.StepLength = 1.0
    problem.FunValDiff = 1e-6
    problem.Print = True
    
    t1 = time.time()
    if optmethod == 'LBFGS':
        opt = PLBFGS(problem)
    if optmethod == 'NLCG':
        opt = PNLCG(problem)
    nflip = 0
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
            print('翻转次数:',nflip)
            break
        else:
            nflip +=1
            problem.x0 = node[isFreeNode].T.flatten()
            mesh = problem.mesh
            if P==1: 
                problem.Preconditioner.reset()

    t2 = time.time()
    print('optimize time:',t2-t1)
    node = mesh.entity('node')
    isFreeNode = ~isFixNode
    n = len(x)//3 
    node[isFreeNode,0] = x[:n]
    node[isFreeNode,1] = x[n:2*n]
    node[isFreeNode,2] = x[2*n:]

    q = TetMeshProblem.get_quality(mesh)
    mesh.celldata['radius_ratio'] = 1/q
    mesh.to_vtk(fname='opt_lshape.vtu')
    angle = mesh.dihedral_angle()
    min_angle = bm.min(angle,axis=1)
    show_angle(min_angle,ylim=5000,title='Radius Ratio Optimize Mesh Dihedral Angle',filename='lshape_opt_angle')

    show_mesh_quality(q,ylim=5000,title='Radius Ratio Optimize Mesh Quality',filename='lshape_opt')
    return mesh

def test_intersectsphere(h=0.08, optmethod='LBFGS', P=0):
    #filename = 'data/12sphere_init.vtu'
    #mesh = TetrahedronMesh.from_vtu(filename)
    mesh = MeshModel.intersect_spheres(h)
    node = mesh.entity('node')
    boundary_node_flag = mesh.boundary_node_flag()
    spherecenter = bm.array([
        [1.0,0.0,0.0],[0.5,0.866025403784439,0.0],[-0.5,0.866025403784439,0.0],
        [-1.0,0.0,0.0],[-0.5,-0.866025403784439,0.0],[0.5,-0.866025403784439,0.0],
        [2.0,0.0,0.0],[1.0,1.73205080756888,0.0],[-1.0,1.73205080756888,0.0],
        [-2.0,0.0,0.0],[-1.0,-1.73205080756888,0.0],[1.0,-1.73205080756888,0.0]
        ],dtype=mesh.ftype)
    NN = node.shape[0]
    ns = spherecenter.shape[0]
    NC = mesh.number_of_cells()
    ylim = int(0.1*NC)
    volume = mesh.entity_measure('cell')
    R = 0.7
    R2 = R * R
    tol = 1e-3
    # 记录每个边界点属于哪个球/哪两个球
    pair_list = bm.array([
    [0,  1],[1,  2],[2,  3],[3,  4],[4,  5],[5,  0],
    [0,  6],[1,  7],[2,  8],[3,  9],[4, 10],[5, 11],
    ], dtype=bm.int32)

    errmat = bm.zeros((NN, ns), dtype=mesh.ftype)
    for i in range(ns):
        diff = node - spherecenter[i]
        dist2 = bm.sum(diff**2, axis=1)
        errmat[:, i] = bm.abs(dist2 - R2)

    face_sphere_id = -bm.ones(NN, dtype=bm.int32)
    edge_pair_id = -bm.ones((NN, 2), dtype=bm.int32)

    face_tol = 8e-3
    pair_tol = 1.6e-2   # 约等于 2*face_tol

    for idx in bm.where(boundary_node_flag)[0]:
        err = errmat[idx]

        # 最佳单球
        s = bm.argmin(err)
        best_sphere_err = err[s]

        # 最佳合法球对
        pair_errs = err[pair_list[:, 0]] + err[pair_list[:, 1]]
        p = bm.argmin(pair_errs)
        j, k = pair_list[p]
        best_pair_err = pair_errs[p]

        # 判为边点的条件：
        # 1) 这对合法球的总误差足够小
        # 2) 两个单独误差都不算大
        # 3) 这两个球确实都比其它球更像
        if (best_pair_err < pair_tol and
            err[j] < face_tol and
            err[k] < face_tol):
            edge_pair_id[idx, 0] = j
            edge_pair_id[idx, 1] = k
        else:
            face_sphere_id[idx] = s

    isBdFaceNode = face_sphere_id >= 0
    isBdEdgeNode = edge_pair_id[:, 0] >= 0
    # 可选：做一次一致性检查
    if bm.any(boundary_node_flag & ~(isBdFaceNode | isBdEdgeNode)):
        print("Warning: some boundary nodes are not classified.")
    if bm.any(isBdFaceNode & isBdEdgeNode):
        print("Warning: some boundary nodes are both face and edge nodes.")
    bad = bm.where(boundary_node_flag & ~(isBdFaceNode | isBdEdgeNode))[0]
    print("unclassified boundary nodes:", len(bad))

    for idx in bad[:10]:
        diff = node[idx] - spherecenter
        dist2 = bm.sum(diff**2, axis=1)
        err = bm.abs(dist2 - R2)
        print(f"idx={idx}, min err={err.min()}, argmin={err.argmin()}, err={err}")    
    
    # 辅助几何函数
    def project_to_circle(points, c1, c2, R=0.7, eps=1e-14):
        """
        将 points 投影到两个等半径球面的交圆上
        points: (N, 3)
        c1, c2: (3,)
        """
        m = 0.5 * (c1 + c2)
        dvec = c2 - c1
        d = bm.linalg.norm(dvec)
        a = dvec / d
        rho = bm.sqrt(R*R - 0.25*d*d)

        v = points - m[None, :]
        v_plane = v - bm.sum(v * a[None, :], axis=1, keepdims=True) * a[None, :]
        nv = bm.linalg.norm(v_plane, axis=1, keepdims=True)

        bad = nv[:, 0] < eps
        if bm.any(bad):
            tmp = bm.array([1.0, 0.0, 0.0], dtype=points.dtype)
            if abs(bm.dot(tmp, a)) > 0.9:
                tmp = bm.array([0.0, 1.0, 0.0], dtype=points.dtype)
            e1 = tmp - bm.dot(tmp, a) * a
            e1 = e1 / bm.linalg.norm(e1)
            v_plane[bad] = e1[None, :]
            nv[bad] = 1.0

        u = v_plane / nv
        return m[None, :] + rho * u

    # 曲面点解析法向
    def normal2d(node):
        normal = bm.zeros_like(node)
        idx = bm.where(isBdFaceNode)[0]
        if len(idx) == 0:
            return normal

        sid = face_sphere_id[idx]
        r = node[idx] - spherecenter[sid]
        nr = bm.linalg.norm(r, axis=1, keepdims=True)
        nr[nr == 0.0] = 1.0
        normal[idx] = r / nr
        return normal

    # 曲边点切向投影
    def tangent_project1d(grad, node):
        idxs = bm.where(isBdEdgeNode)[0]
        for idx in idxs:
            j, k = edge_pair_id[idx]
            c1 = spherecenter[j]
            c2 = spherecenter[k]

            a = c2 - c1
            a = a / bm.linalg.norm(a)

            m = 0.5 * (c1 + c2)
            v = node[idx] - m
            v = v - bm.dot(v, a) * a
            nv = bm.linalg.norm(v)
            if nv == 0.0:
                continue
            v = v / nv

            t = bm.cross(a, v)
            nt = bm.linalg.norm(t)
            if nt == 0.0:
                continue
            t = t / nt

            grad[idx] = bm.dot(grad[idx], t) * t
        return grad

    # 边界投影
    def project_to_boundary(node):
        # 1) 曲面点：投到所属球面
        idxs = bm.where(isBdFaceNode)[0]
        if len(idxs) > 0:
            sid = face_sphere_id[idxs]
            r = node[idxs] - spherecenter[sid]
            nr = bm.linalg.norm(r, axis=1, keepdims=True)
            nr[nr == 0.0] = 1.0
            node[idxs] = spherecenter[sid] + R * r / nr

        # 2) 曲边点：投到所属两球交圆
        idxs = bm.where(isBdEdgeNode)[0]
        for idx in idxs:
            j, k = edge_pair_id[idx]
            node[idx:idx+1] = project_to_circle(
                node[idx:idx+1], spherecenter[j], spherecenter[k], R=R
            )

        return node

    q = TetMeshProblem.get_quality(mesh)
    angle = mesh.dihedral_angle()
    min_angle = bm.min(angle,axis=1)
    show_angle(min_angle,ylim=18000,title='Initial Mesh Dihedral Angle',filename='tsp_init_angle')

    show_mesh_quality(q,ylim=18000,title='Initial Mesh Quality',filename='tsp_init') 

    options = TetMeshProblem.get_options(
            mesh=mesh,
            isBdEdgeNode = isBdEdgeNode,
            isBdFaceNode = isBdFaceNode,
            Project = project_to_boundary,
            Tangent1d=tangent_project1d,
            Normal2d = normal2d)
    problem = TetMeshProblem(options)

    NDof = len(problem.x0)
    if P==1:
        problem.Preconditioner = problem.build_preconditioner(
                update_interval=3,
                rtol=1e-2,
                maxiter=100)
    elif P==0:
        problem.Preconditioner = None
    problem.StepLength = 1.0
    problem.FunValDiff = 1e-6
    problem.Print = True

    t1 = time.time()
    if optmethod == 'LBFGS':
        opt = PLBFGS(problem)
    if optmethod == 'NLCG':
        opt = PNLCG(problem)
    nflip=0
    while True:
        x, f, g = opt.run()
        node = mesh.entity('node')
        n = len(x)//3 
        node[:,0] = x[:n]
        node[:,1] = x[n:2*n]
        node[:,2] = x[2*n:]
        problem.mesh = mesh
        if not problem.flipopt():
            print('翻转次数:',nflip)
            break
        else:
            nflip +=1
            problem.x0 = node.T.flatten()
            mesh = problem.mesh
            if P==1: 
                problem.Preconditioner.reset()


    node = mesh.entity('node')
    n = len(x)//3 
    node[:,0] = x[:n]
    node[:,1] = x[n:2*n]
    node[:,2] = x[2*n:]
    t2 = time.time()
    print('time:',t2-t1)

    q = TetMeshProblem.get_quality(mesh)
    mesh.celldata['radius_ratio'] = 1/q
    mesh.to_vtk(fname='opt_tsp.vtu')
    angle = mesh.dihedral_angle()
    min_angle = bm.min(angle,axis=1)
    show_angle(min_angle,ylim=18000,title='Radius Ratio Optimize Mesh Dihedral Angle',filename='tsp_opt_angle')
    show_mesh_quality(q,ylim=18000,title='Radius Ratio Optimize Mesh Quality',filename='tsp_opt')
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

def show_angle(q1,ylim=8000,title=None,filename=None):
    fig,axes= plt.subplots()
    q1 = bm.to_numpy(q1)
    minq1 = np.min(q1)
    maxq1 = np.max(q1)
    meanq1 = np.mean(q1)
    rmsq1 = np.sqrt(np.mean(q1**2))
    stdq1 = np.std(q1)
    NC = len(q1)
    SNC = np.sum((q1<30))
    hist, bins = np.histogram(q1, bins=50, range=(0, 71))
    center = (bins[:-1] + bins[1:]) / 2
    width = bins[1]-bins[0]
    axes.bar(center, hist, align='center', width=0.9*width)
    axes.set_xlim(0, 71)
    axes.set_ylim(0,ylim)

    if title is not None:
        axes.set_title(title, fontsize=16, pad=20)

    #TODO: fix the textcoords warning
    axes.annotate('Min dihedral angle:{:.6}'.format(minq1), xy=(0, 0),
            xytext=(0.15, 0.85),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    axes.annotate('Max dihedral angle: {:.6}'.format(maxq1), xy=(0, 0),
            xytext=(0.15, 0.8),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    axes.annotate('Average dihedral angle: {:.6}'.format(meanq1), xy=(0, 0),
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

    axes.annotate('Dihedral angle less than 30:{:.0f}/{:.0f}'.format(SNC,NC), xy=(0, 0),
            xytext=(0.15, 0.6),
            textcoords="figure fraction",
            horizontalalignment='left', verticalalignment='top', fontsize=15)
    plt.tight_layout() 
    filename = filename+f"p_{args.p}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return 0

# main function to run the tests
if args.exam == 'sp':
    mesh = test_unit_sphere(optmethod=args.optmethod, P=args.p)
if args.exam == 'ls':
    mesh = test_LShape(optmethod=args.optmethod, P=args.p)
if args.exam == 'tsp':
    mesh = test_intersectsphere(optmethod=args.optmethod, P=args.p)

