import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.opt import PLBFGS
from fealpy.mesh import TetrahedronMesh
from fealpy.meshopt import RadiusRatioQuality
from fealpy.meshopt import RadiusRatioSumObjective
from fealpy.operator import LinearOperator

parser = argparse.ArgumentParser(description=
        '''
        半径比优化示例
        ''')

parser.add_argument('--p', 
            default=1,type=int,
            help='预处理器类型，0表示不使用预处理器，1表示使用预处理器')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='默认后端为numpy')

args = parser.parse_args()
bm.set_backend(args.backend)

def test_sphere(P=0):
    mesh = TetrahedronMesh.from_vtu('../../data/sphere_tetmesh.vtu')
    b = mesh.entity_barycenter("cell")
    
    mesh_quality = RadiusRatioQuality(mesh)
    sum_q = RadiusRatioSumObjective(mesh_quality)
    q = mesh_quality(mesh.node)
    NC = mesh.number_of_cells()
    
    show_mesh_quality(bm.to_numpy(q),ylim=2500) 

    NDof = len(sum_q.x0)
    if P==1:
        Preconditioner = LinearOperator((NDof,NDof),sum_q.preconditioner)
        update_Preconditioner = sum_q.update_preconditioner
    elif P==0:
        Preconditioner = None
        update_Preconditioner = None
    x0 = sum_q.x0
    options = PLBFGS.get_options(
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
    '''
    node[isFreeNode,0] = x[:n]
    node[isFreeNode,1] = x[n:2*n]
    node[isFreeNode,2] = x[2*n:]
    '''
    mesh_quality = RadiusRatioQuality(mesh)
    q = mesh_quality(node) 

    show_mesh_quality(bm.to_numpy(q),ylim=2500)
    return mesh

def show_mesh_quality(q1,ylim=8000):
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
    plt.show()
    return 0

mesh = test_sphere(P=args.p)

