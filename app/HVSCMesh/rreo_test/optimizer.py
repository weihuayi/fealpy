import numpy as np
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.mesh.tetrahedron_mesh import TetrahedronMesh
from mesh_quality import RadiusRatioQuality
from radius_ratio_objective import RadiusRatioSumObjective

def show_mesh_quality(q1,ylim=1000):
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

def show_angle(axes, angle):
        """
        @brief 显示网格角度的分布直方图
        """
        hist, bins = np.histogram(angle.flatten('F') * 180 / bm.pi, bins=50, range=(0, 180))
        center = (bins[:-1] + bins[1:]) / 2
        axes.bar(center, hist, align='center', width=180 / 50.0)
        axes.set_xlim(0, 180)
        mina = np.min(angle.flatten('F') * 180 / np.pi)
        maxa = np.max(angle.flatten('F') * 180 / np.pi)
        meana = np.mean(angle.flatten('F') * 180 / np.pi)
        axes.annotate('Min angle: {:.4}'.format(mina), xy=(0.41, 0.5),
                      xytext=(0.15,0.85),
                      textcoords='axes fraction',
                      horizontalalignment='left', verticalalignment='top')
        axes.annotate('Max angle: {:.4}'.format(maxa), xy=(0.41, 0.45),
                      xytext=(0.15,0.75),
                      textcoords='axes fraction',
                      horizontalalignment='left', verticalalignment='top')
        axes.annotate('Average angle: {:.4}'.format(meana), xy=(0.41, 0.40),
                      xytext=(0.15,0.7),
                      textcoords='axes fraction',
                      horizontalalignment='left', verticalalignment='top')
        return mina, maxa, meana

def BlockJacobi2d(node,A,B,isFreeNode):
    NN = node.shape[0] 
    isBdNode = ~isFreeNode
    newNode = bm.zeros((NN, 2), dtype=bm.float64)
    newNode[isBdNode, :] = node[isBdNode, :]        
    b = -B*node[:, 1] - A*newNode[:, 0]
    newNode[isFreeNode, 0], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
            b[isFreeNode], x0=node[isFreeNode, 0], tol=1e-6)
    b = B*node[:, 0] - A*newNode[:, 1]
    newNode[isFreeNode, 1], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
            b[isFreeNode], x0=node[isFreeNode, 1], tol=1e-6)
    node[isFreeNode, :] = newNode[isFreeNode, :]
    return node

def BlockJacobi3d(node,A,B0,B1,B2,isFreeNode):
    NN = node.shape[0]
    isBdNode = ~isFreeNode
    newNode = bm.zeros((NN, 3), dtype=bm.float64)
    newNode[isBdNode, :] = node[isBdNode, :]
    b = -B2*node[:, 1] -B1*node[:, 2] - A*newNode[:, 0]
    newNode[isFreeNode, 0], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
            b[isFreeNode], x0=node[isFreeNode, 0], tol=1e-6)
    b = B2*node[:, 0] - A*newNode[:, 1] - B0*node[:, 2]
    newNode[isFreeNode, 1], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
            b[isFreeNode], x0=node[isFreeNode, 1], rtol=1e-6)
    b = B1*node[:,0]+B0*node[:,1]-A*newNode[:,2]
    newNode[isFreeNode, 2], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
            b[isFreeNode], x0=node[isFreeNode, 2], rtol=1e-6)
    #node[isFreeNode, :] = newNode[isFreeNode, :]
    p = bm.zeros((NN,3),dtype=bm.float64)
    p[isFreeNode,:] = newNode[isFreeNode,:] - node[isFreeNode,:]
    #node += 0.7*p
    node += 0.3*p
    return node

def iterate_solver(mesh,funtype=0):
    NC = mesh.number_of_cells()
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    isFreeNode = ~mesh.boundary_node_flag()
    q = bm.zeros((2, NC),dtype=bm.float64)
    mesh_quality = RadiusRatioQuality(mesh)
    mesh_objective = RadiusRatioSumObjective(mesh_quality)
    q[0] = mesh_quality(node)
    minq = bm.min(q[0])
    avgq = bm.mean(q[0])
    maxq = bm.max(q[0])
    print('iter=',0,'minq=',minq,'avgq=',avgq, 'maxq=',maxq)
    if mesh.TD == 2:
        for i in range(0,30):
            A,B = mesh_objective.hess(node,funtype)
            node = BlockJacobi2d(node, A, B, isFreeNode)
                        ## count quality
            q[1] = mesh_quality(node)
            minq = bm.min(q[1])
            avgq = bm.mean(q[1])
            maxq = bm.max(q[1])
            print('iter=',i+1,'minq=',minq,'avgq=',avgq, 'maxq=',maxq)
            
            if bm.max(np.abs(q[1]-q[0]))<1e-8:
                print("Bjacobi迭代次数为%d次"%(i+1))
                break
            q[0] = q[1]
            
        mesh = TriangleMesh(node,cell)
    elif mesh.TD == 3:
        for i in range(0,100):
            A,B0,B1,B2 = mesh_objective.hess(node,funtype)
            node = BlockJacobi3d(node, A, B0, B1, B2, isFreeNode)
                        ## count quality
            q[1] = mesh_quality(node)
            minq = bm.min(q)
            avgq = bm.mean(q)
            maxq = bm.max(q)
            print('minq=',minq,'avgq=',avgq, 'maxq=',maxq)

            if bm.max(np.abs(q[1]-q[0]))<1e-8:
                print("Bjacobi迭代次数为%d次"%(i+1))
                break
            q[0] = q[1]

        mesh = TetrahedronMesh(node,cell)
    return mesh
