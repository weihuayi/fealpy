#!/usr/bin/python
# Filename: TriangleOpt.py
# Author: Huayi Wei

import vtk
from vtk.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, spdiags, triu, tril, find, hstack, eye
from scipy.sparse.linalg import cg, inv, dsolve

from scipy.linalg import norm
from pyamg import *

from matplotlib.tri import Triangulation
import time

def GetAngle(points, cells, cellLocations):
    NT = cellLocations.shape[0]
    idxi = cells[cellLocations + 1]
    idxj = cells[cellLocations + 2]
    idxk = cells[cellLocations + 3]

    v0 = points[np.ix_(idxj, [0, 1])] - points[np.ix_(idxi, [0, 1])]
    v1 = points[np.ix_(idxk, [0, 1])] - points[np.ix_(idxi, [0, 1])]
    angle = np.zeros((NT, 3), dtype=np.float)
    angle[:, 0] = np.arccos(np.sum(v0*v1, axis=1)/np.sqrt(np.sum(v0**2, axis=1)
        *np.sum(v1**2, axis=1)))
                                    
    v0 = points[np.ix_(idxk, [0, 1])] - points[np.ix_(idxj, [0, 1])]
    v1 = points[np.ix_(idxi, [0, 1])] - points[np.ix_(idxj, [0, 1])]
    angle[:, 1] = np.arccos(np.sum(v0*v1, axis=1)/np.sqrt(np.sum(v0**2, axis=1)
        *np.sum(v1**2, axis=1)))

    v0 = points[np.ix_(idxi, [0, 1])] - points[np.ix_(idxk, [0, 1])]
    v1 = points[np.ix_(idxj, [0, 1])] - points[np.ix_(idxk, [0, 1])]
    angle[:, 2] = np.arccos(np.sum(v0*v1, axis=1)/np.sqrt(np.sum(v0**2, axis=1)
        *np.sum(v1**2, axis=1)))

    return angle


def GetFreeNodeInfo(N, cells, cellLocations):
    NT = cellLocations.shape[0]
    idx = np.zeros((3*NT, 2), dtype=np.int)
    next = [2, 3, 1]
    prev = [3, 1, 2]
    for i in range(0, 3):
        idx[i*NT:(i+1)*NT, 0] = cells[cellLocations+next[i]]
        idx[i*NT:(i+1)*NT, 1] = cells[cellLocations+prev[i]]
    idx.sort()
    I = np.ones((3*NT, ), dtype=np.int)
    A = csr_matrix((I, (idx[:, 0], idx[:, 1])), shape=(N, N))
    NE = A.getnnz()
    edge = np.zeros((NE, 2), dtype=np.int)
    edge[:, 0], edge[:, 1], val = find(A)
    isBdEdge = val == 1

    #import pdb
    #pdb.set_trace()
    isFreeNode = np.ones((N, ), dtype=np.bool)
    isFreeNode[edge[isBdEdge, 0]] = False
    isFreeNode[edge[isBdEdge, 1]] = False

    return isFreeNode


def GetQuality(points, cells, cellLocations):
    idxi = cells[cellLocations+1]
    idxj = cells[cellLocations+2]
    idxk = cells[cellLocations+3]
    v0 = points[idxk, :] - points[idxj, :]
    v1 = points[idxi, :] - points[idxk, :]
    v2 = points[idxj, :] - points[idxi, :]

    NT = cellLocations.shape[0]
    l = np.zeros((NT, 3))
    l[:, 0] = np.sqrt(np.sum(v0**2, axis=1))
    l[:, 1] = np.sqrt(np.sum(v1**2, axis=1))
    l[:, 2] = np.sqrt(np.sum(v2**2, axis=1))
    p = l.sum(axis=1)
    q = l.prod(axis=1)
    a = 0.5*(-v2[:, 0]*v1[:, 1] + v2[:, 1]*v1[:, 0])

    quality = 16*a**2/(p*q)

    return quality


def ShowVTKMeshQuality(mesh, ax):
    wmesh = dsa.WrapDataObject(mesh)
    points = wmesh.GetPoints()
    cells = wmesh.GetCells()
    cellLocations = wmesh.GetCellLocations()
    ShowMeshQuality(points, cells, cellLocations, ax)
    return


def ShowMeshQuality(points, cells, cellLocations, ax, quality=None):
    if quality is None:
        quality = GetQuality(points, cells, cellLocations)
    minq = np.min(quality)
    maxq = np.max(quality)
    meanq = np.mean(quality)
    hist, bins = np.histogram(quality, bins=50, range=(0, 1))
    center = (bins[:-1] + bins[1:]) / 2
    ax.bar(center, hist, align='center', width=0.02)
    ax.set_xlim(0, 1)
    ax.annotate('Min quality: {:.6}'.format(minq), xy=(0.1, 0.5),
            textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    ax.annotate('Max quality: {:.6}'.format(maxq), xy=(0.1, 0.45),
            textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    ax.annotate('Average quality: {:.6}'.format(meanq), xy=(0.1, 0.40),
            textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    return


def ShowVTKMeshAngle(mesh, ax):
    wmesh = dsa.WrapDataObject(mesh)
    points = wmesh.GetPoints()
    cells = wmesh.GetCells()
    cellLocations = wmesh.GetCellLocations()
    ShowMeshAngle(points, cells, cellLocations, ax)
    return 


def ShowMeshAngle(points, cells, cellLocations, ax, angle=None):
    if angle is None:
        angle = GetAngle(points, cells, cellLocations)

    hist, bins = np.histogram(angle.flatten('F')*180/np.pi, bins=50, range=(0, 180))
    center = (bins[:-1] + bins[1:])/2
    ax.bar(center, hist, align='center', width=180/50.0)
    ax.set_xlim(0, 180)
    mina = np.min(angle.flatten('F')*180/np.pi)
    maxa = np.max(angle.flatten('F')*180/np.pi)
    meana = np.mean(angle.flatten('F')*180/np.pi)
    ax.annotate('Min angle: {:.4}'.format(mina), xy=(0.41, 0.5),
            textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    ax.annotate('Max angle: {:.4}'.format(maxa), xy=(0.41, 0.45),
            textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    ax.annotate('Average angle: {:.4}'.format(meana), xy=(0.41, 0.40),
            textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='top')
    return


def ShowVTKMesh(mesh, ax):
    wmesh = dsa.WrapDataObject(mesh)

    points = wmesh.GetPoints()
    cells = wmesh.GetCells()
    cellLocations = wmesh.GetCellLocations()
    N = points.shape[0]
    NT = cellLocations.shape[0]
    triangles = cells.reshape((NT, 4))[:, 1:]
    tri = Triangulation(points[:, 0], points[:, 1], triangles)
    ax.set_aspect('equal')
    ax.triplot(tri)
    ax.set_axis_off()
    return


class vtkTriangleRadiusRatioOpt(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self,
                nInputPorts=1, inputType='vtkUnstructuredGrid',
                nOutputPorts=1, outputType='vtkUnstructuredGrid')

        self.__MaxIt = 20
        self.__ItMethod = 'BJ'

    def RequestData(self, request, inInfo, outInfo):
        inp = vtk.vtkUnstructuredGrid.GetData(inInfo[0])
        opt = vtk.vtkUnstructuredGrid.GetData(outInfo)
        opt.ShallowCopy(inp)

        wopt = dsa.WrapDataObject(opt)
        points = wopt.GetPoints()
        cells = wopt.GetCells()
        cellLocations = wopt.GetCellLocations()

        N = points.shape[0]
        isFreeNode = GetFreeNodeInfo(N, cells, cellLocations)
        quality = GetQuality(points, cells, cellLocations)
        minq = np.min(quality)
        avgq = np.mean(quality)
        print 0, minq, avgq
        for i in range(0, self.__MaxIt):
            A, B = self.GetIterateMatrix(points, cells, cellLocations)
            if self.__ItMethod == 'BJ':
                self.BlockJacobi(points, A, B, isFreeNode)
            elif self.__ItMethod == 'BG':
                self.BlockGauss(points, A, B, isFreeNode)
            elif self.__ItMethod == 'J':
                self.Jacobi(points, A, B, isFreeNode)
            else:
                print 'I do not know your method!'
                return 0
            quality = GetQuality(points, cells, cellLocations)
            minq = np.min(quality)
            avgq = np.mean(quality)
            print i+1, minq, avgq
        return 1

    def Jacobi(self, points, A, B, isFreeNode):
        N = points.shape[0]
        D = spdiags(1.0/A.diagonal(), 0, N, N)
        C = -(triu(A, 1) + tril(A, -1))
        X = D*(C*points[:, 0] - B*points[:, 1])
        Y = D*(B*points[:, 0] + C*points[:, 1])
        points[isFreeNode, 0] = X[isFreeNode]
        points[isFreeNode, 1] = Y[isFreeNode]
        return

    def BlockJacobi(self, points, A, B, isFreeNode):
        N = points.shape[0]
        isBdNode = np.logical_not(isFreeNode)
        newPoints = np.zeros((N, 3), dtype=np.float)
        newPoints[isBdNode, :] = points[isBdNode, :]
        
        b = -B*points[:, 1] - A*newPoints[:, 0]
        #t0 = time.time()
        newPoints[isFreeNode, 0], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode], x0=points[isFreeNode, 0], tol=1e-6)
        #print time.time() - t0


        b = B*points[:, 0] - A*newPoints[:, 1]
        #t0 = time.time()
        newPoints[isFreeNode, 1], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode], x0=points[isFreeNode, 1], tol=1e-6)
        #print time.time() - t0

        points[isFreeNode, :] = newPoints[isFreeNode, :]
        return

    def BlockGauss(self, points, A, B, isFreeNode):
        N = points.shape[0]
        isBdNode = np.logical_not(isFreeNode)
        newPoints = np.zeros((N, 2), dtype=np.float)
        newPoints[isBdNode, :] = points[np.ix_(isBdNode, [0, 1])]
        ml = smoothed_aggregation_solver(A[np.ix_(isFreeNode, isFreeNode)])
        M = ml.aspreconditioner(cycle='W')
        b = -B*points[:, 1] - A*newPoints[:, 0]
        points[isFreeNode, 0], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode], x0=points[isFreeNode, 0], tol=1e-8, M=M)
        b = B*points[:, 0] - A*newPoints[:, 1]
        points[isFreeNode, 1], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode], x0=points[isFreeNode, 1], tol=1e-8, M=M)
        return

    def SetMaxIt(self, maxIt):
        if maxIt != self.__MaxIt:
            self.Modified()
            self.__MaxIt = maxIt

    def GetMaxIt(self):
        return self.__MaxIt

    def SetItMethod(self, method):
        if method != self.__ItMethod:
            self.Modified()
            self.__ItMethod = method

    def GetItMethod(self):
        return self.__ItMethod
   
    def GetIterateMatrix(self, points, cells, cellLocations):
        N = points.shape[0]
        NT = cellLocations.shape[0]

        idxi = cells[cellLocations + 1]
        idxj = cells[cellLocations + 2]
        idxk = cells[cellLocations + 3]

        v0 = points[np.ix_(idxk, [0, 1])] - points[np.ix_(idxj, [0, 1])]
        v1 = points[np.ix_(idxi, [0, 1])] - points[np.ix_(idxk, [0, 1])]
        v2 = points[np.ix_(idxj, [0, 1])] - points[np.ix_(idxi, [0, 1])]

        area = 0.5*(-v2[:, [0]]*v1[:, [1]] + v2[:, [1]]*v1[:, [0]])
        l2 = np.zeros((NT, 3), dtype=np.float)
        l2[:, 0] = np.sum(v0**2, axis=1)
        l2[:, 1] = np.sum(v1**2, axis=1)
        l2[:, 2] = np.sum(v2**2, axis=1)
        l = np.sqrt(l2)
        p = l.sum(axis=1, keepdims=True)
        q = l.prod(axis=1, keepdims=True)
        mu = p*q/(16*area**2)
        c = mu*(1/(p*l) + 1/l2)
        val = np.concatenate((
            c[:, [1, 2]].sum(axis=1), -c[:, 2], -c[:, 1],
            -c[:, 2], c[:, [0, 2]].sum(axis=1), -c[:, 0],
            -c[:, 1], -c[:, 0], c[:, [0, 1]].sum(axis=1)))
        I = np.concatenate((
            idxi, idxi, idxi,
            idxj, idxj, idxj,
            idxk, idxk, idxk))
        J = np.concatenate((idxi, idxj, idxk))
        J = np.concatenate((J, J, J))
        A = csr_matrix((val, (I, J)), shape=(N, N))

        cn = mu/area
        cn.shape = (cn.shape[0],)
        val = np.concatenate((-cn, cn, cn, -cn, -cn, cn))
        I = np.concatenate((idxi, idxi, idxj, idxj, idxk, idxk))
        J = np.concatenate((idxj, idxk, idxi, idxk, idxi, idxj))
        B = csr_matrix((val, (I, J)), shape=(N, N))

        return (A, B)


class vtkTriangleLinearFEMOpt(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self,
                nInputPorts=1, inputType='vtkUnstructuredGrid',
                nOutputPorts=1, outputType='vtkUnstructuredGrid')

    def RequestData(self, request, inInfo, outInfo):
        inp = vtk.vtkUnstructuredGrid.GetData(inInfo[0])
        opt = vtk.vtkUnstructuredGrid.GetData(outInfo)
        opt.ShallowCopy(inp)

        wopt = dsa.WrapDataObject(opt)
        points = wopt.GetPoints()
        cells = wopt.GetCells()
        cellLocations = wopt.GetCellLocations()

        quality = GetQuality(points, cells, cellLocations)
        minq = np.min(quality)
        avgq = np.mean(quality)
        print 0, minq, avgq

        N = points.shape[0]
        isFreeNode = GetFreeNodeInfo(N, cells, cellLocations)
        isBdNode = np.logical_not(isFreeNode)
        A = self.GetStiffMatrix(points, cells, cellLocations)
        newPoints = np.zeros((N, 2), dtype=np.float)
        #import pdb
        #pdb.set_trace()
        newPoints[isBdNode, :] = points[np.ix_(isBdNode, [0, 1])]
        ml = smoothed_aggregation_solver(A[np.ix_(isFreeNode, isFreeNode)]) 
        M = ml.aspreconditioner(cycle='W')
        b = -A*newPoints
        points[isFreeNode, 0], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode, 0], x0=points[isFreeNode, 0], tol=1e-8, M=M)
        points[isFreeNode, 1], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
                b[isFreeNode, 1], x0=points[isFreeNode, 1], tol=1e-8, M=M)

        quality = GetQuality(points, cells, cellLocations)
        minq = np.min(quality)
        avgq = np.mean(quality)
        print 0, minq, avgq
        return 1

    def GetIdeaAngle(self, points, cells, cellLocations):
        NT = cellLocations.shape[0]
        N = points.shape[0]
        angle = GetAngle(points, cells, cellLocations)
        idx = cells.reshape((NT, 4))[:, 1:]
        sumAngle = np.bincount(idx.flatten(), angle.flatten())
        valence = np.bincount(idx.flatten())
        ideaAngle = sumAngle/valence

        IA = ideaAngle[idx].reshape((3*NT, 1), order='F')
        #IA = np.pi/3*np.ones((3*NT, 1), dtype=np.float)
        val = np.ones((NT,), dtype=np.float)
        J = np.arange(0, NT)
        C1 = csc_matrix((val, (cells[cellLocations+1], J)), shape=(N, NT))
        C2 = csc_matrix((val, (cells[cellLocations+2], J)), shape=(N, NT))
        C3 = csc_matrix((val, (cells[cellLocations+3], J)), shape=(N, NT))
        C = hstack([C1, C2, C3])
        B = hstack([eye(NT), eye(NT), eye(NT)])
        P = C*B.transpose()
        D = C*C.transpose()
        Q = P*P.transpose()
        G = 3*D - Q
        F = np.pi*np.ones((NT, 1), dtype=np.float)
        U = F - B*IA
        H = 2*P*U
        Gamma2 = np.zeros((N, 1), dtype=np.float)
        
        isFreeNode = np.ones((N, ), dtype=np.bool)
        isFreeNode[0] = True

        print type(np.array(H[isFreeNode].tolist(), dtype=np.float))
        residuals = []
        ml = ruge_stuben_solver(G[np.ix_(isFreeNode, isFreeNode)]) 
        Gamma2[isFreeNode] = ml.solve(np.array(H[isFreeNode].tolist(),
            dtype=np.float), tol=1e-6, accel='cg', residuals=residuals)
        #Gamma2[isFreeNode] = inv(G[np.ix_(isFreeNode, isFreeNode)].tocsc())*H[isFreeNode]
        Gamma1 = -2.0/3.0*U - 1.0/3.0*P.transpose()*Gamma2
        A = IA - 0.5*B.transpose()*Gamma1 - 0.5*C.transpose()*Gamma2
        return A.reshape((NT, 3), order='F')

    def GetStiffMatrix(self, points, cells, cellLocations):
        N = points.shape[0]
        ideaAngle = self.GetIdeaAngle(points, cells, cellLocations)
        idxi = cells[cellLocations+1]
        idxj = cells[cellLocations+2]
        idxk = cells[cellLocations+3]

        Ai = np.cos(ideaAngle[:, 0])/np.sin(ideaAngle[:, 0])/2.0
        Aj = np.cos(ideaAngle[:, 1])/np.sin(ideaAngle[:, 1])/2.0
        Ak = np.cos(ideaAngle[:, 2])/np.sin(ideaAngle[:, 2])/2.0
        val = np.concatenate((
            Aj+Ak, -Ak, -Aj,
            -Ak, Ai+Ak, -Ai,
            -Aj, -Ai, Ai+Aj))
        I = np.concatenate((
            idxi, idxi, idxi,
            idxj, idxj, idxj,
            idxk, idxk, idxk))
        J = np.concatenate((idxi, idxj, idxk))
        J = np.concatenate((J, J, J))
        A = csr_matrix((val, (I, J)), shape=(N, N))

        return A
