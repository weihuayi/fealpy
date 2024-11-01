import sys
import numpy as np
import mpl_toolkits.mplot3d as a3
import numpy.matlib

import pylab as pl

from mayavi import mlab

from fealpy.mesh.surface_mesh_generator import iso_surface
from fealpy.mesh.level_set_function import Sphere
from fealpy.mesh.TriangleMesh import TriangleMesh 

surface = Sphere()
#mesh = surface.init_mesh()
#mesh.uniform_refine(2, surface)

n = 10
mesh = iso_surface(surface, surface.box, nx=n, ny=n, nz=n)


point = mesh.point
n1 = point.shape[0]

cell = mesh.ds.cell
normal = surface.unit_normal(point)

def updatecircumcenter(point, cell):
    v0 = point[cell[:, 0], :]
    v1 = point[cell[:, 1], :]
    v2 = point[cell[:, 2], :]

    v10 = v1 - v0
    v20 = v2 - v0
    v21 = v2 - v1

    L10 = np.sum(v10 ** 2, 1).reshape(-1, 1)
    L20 = np.sum(v20 ** 2, 1).reshape(-1, 1)
    L21 = np.sum(v21 ** 2, 1).reshape(-1, 1)

    flag0 = np.array(L10 + L20 < L21, dtype=np.int)
    flag1 = np.array(L21 + L10 < L20, dtype=np.int)
    flag2 = np.array(L20 + L21 < L10, dtype=np.int)

    c, R = mesh.circumcenter()

    c[flag0, :] = (v1[flag0, :] + v2[flag0, :]) / 2
    c[flag1, :] = (v0[flag1, :] + v2[flag1, :]) / 2
    c[flag2, :] = (v0[flag2, :] + v1[flag2, :]) / 2
    return c

def curvaturedensity(point,cell,normal,gamma=1):
    n = normal[cell[:,0],:] + normal[cell[:,1],:] + normal[cell[:,2],:]
    v01 = point[cell[:,1],:] - point[cell[:,0],:]
    v02 = point[cell[:,2],:] - point[cell[:,0],:]
    
    nv = np.cross(v01,v02,1)
    length = np.sqrt(np.square(nv).sum(axis=1))
    area = length/2.0

    nv = nv/length.reshape((-1, 1))

    cellRho = np.abs(9-np.sum(n**2,1))/area
    cellRho = cellRho + np.sqrt(np.spacing(1))

    for i in range(12):
        sub = cell.flatten(0)
        cellRho_area = cellRho*area
        val = np.matlib.repmat(cellRho_area, 3, 1).flatten(0)
        pointRho = np.bincount(sub, weights=val, minlength=point.shape[0])

        sarea = np.matlib.repmat(area, 3, 1).flatten(0)
        b = np.bincount(sub, weights=sarea, minlength=point.shape[0])
        pointRho = pointRho/b
        cellRho = (pointRho[cell[:,0]] + pointRho[cell[:,1]] + pointRho[cell[:,2]])/3
    cur = pointRho
    pointRho = (cellRho/np.max(cellRho))**(gamma)
    return pointRho, cellRho, cur, area


def MassCenter(point,cell,normal,gamma=1):
    N = point.shape[0]
    NC = cell.shape[0]
    print(NC)
    b = np.zeros((N, 1), dtype=float)
    ax = np.zeros((N, 1), dtype=float)
    ay = np.zeros((N, 1), dtype=float)
    az = np.zeros((N, 1), dtype=float)
    c,R = mesh.circumcenter()
    c1 = updatecircumcenter(point, cell)

    pointRho, cellRho, cur, area = curvaturedensity(point, cell, normal, gamma=1)

    rho = 1

    mc0 = (point[cell[:, 1], :] + point[cell[:, 2], :]) / 2
    mc1 = (point[cell[:, 2], :] + point[cell[:, 0], :]) / 2
    mc2 = (point[cell[:, 0], :] + point[cell[:, 1], :]) / 2

    # first vertex

    tmpbc0 = (mc1 + c + point[cell[:, 0], :])/3
    tmpv0 = np.cross(c - point[cell[:, 0], :], mc1 - point[cell[:, 0], :], 1)
    tmpArea0 = 0.5*np.sqrt(np.sum(tmpv0**2, 1))*rho

    tmpbc1 = (mc2 + c + point[cell[:, 0], :]) / 3
    tmpv1 = np.cross(mc2 - point[cell[:, 0], :], c - point[cell[:, 0], :], 1)
    tmpArea1 = 0.5 * np.sqrt(np.sum(tmpv1 ** 2, 1)) * rho

    sub0 = cell[:, 0]
    val = tmpArea0 + tmpArea1
    b = b + np.bincount(sub0, weights=val, minlength=point.shape[0]).reshape(-1, 1)

    val0 = tmpbc0[:, 0]*tmpArea0 + tmpbc1[:, 0]*tmpArea1
    ax = ax + np.bincount(sub0, val0,minlength=point.shape[0]).reshape(-1, 1)

    val1 = tmpbc0[:, 1]*tmpArea0 + tmpbc1[:, 1]*tmpArea1
    ay = ay + np.bincount(sub0, val1, minlength=point.shape[0]).reshape(-1, 1)

    val2 = tmpbc0[:, 2]*tmpArea0 + tmpbc1[:, 2]*tmpArea1
    az = az + np.bincount(sub0, val2, minlength=point.shape[0]).reshape(-1, 1)
 


    # second vertex

    tmpbc0 = (mc2 + c + point[cell[:, 1], :]) / 3
    tmpv0 = np.cross(c - point[cell[:, 1], :], mc1 - point[cell[:, 1], :], 1)
    tmpArea0 = 0.5 * np.sqrt(np.sum(tmpv0 ** 2, 1)) * rho

    tmpbc1 = (mc0 + c + point[cell[:, 1], :]) / 3
    tmpv1 = np.cross(mc0 - point[cell[:, 1], :], c - point[cell[:, 1], :], 1)
    tmpArea1 = 0.5 * np.sqrt(np.sum(tmpv1 ** 2, 1)) * rho

    sub1 = cell[:, 1]
    val = tmpArea0 + tmpArea1
    b = b + np.bincount(sub1, weights=val, minlength=point.shape[0]).reshape(-1, 1)

    val0 = tmpbc0[:, 0] * tmpArea0 + tmpbc1[:, 0] * tmpArea1
    ax = ax + np.bincount(sub1, val0, minlength=point.shape[0]).reshape(-1, 1)

    val1 = tmpbc0[:, 1] * tmpArea0 + tmpbc1[:, 1] * tmpArea1
    ay = ay + np.bincount(sub1, val1, minlength=point.shape[0]).reshape(-1, 1)

    val2 = tmpbc0[:, 2] * tmpArea0 + tmpbc1[:, 2] * tmpArea1
    az = az + np.bincount(sub1, val2, minlength=point.shape[0]).reshape(-1, 1)

    # third vertex

    tmpbc0 = (mc0 + c + point[cell[:, 2], :]) / 3
    tmpv0 = np.cross(c - point[cell[:, 2], :], mc0 - point[cell[:, 2], :], 1)
    tmpArea0 = 0.5 * np.sqrt(np.sum(tmpv0 ** 2, 1)) * rho

    tmpbc1 = (mc1 + c + point[cell[:, 2], :]) / 3
    tmpv1 = np.cross(mc1 - point[cell[:, 2], :], c - point[cell[:, 2], :], 1)
    tmpArea1 = 0.5 * np.sqrt(np.sum(tmpv1 ** 2, 1)) * rho

    sub2 = cell[:, 2]
    val = tmpArea0 + tmpArea1
    b = b + np.bincount(sub1, weights=val, minlength=point.shape[0]).reshape(-1, 1)

    val0 = tmpbc0[:, 0] * tmpArea0 + tmpbc1[:, 0] * tmpArea1
    ax = ax + np.bincount(sub2, val0, minlength=point.shape[0]).reshape(-1, 1)

    val1 = tmpbc0[:, 1] * tmpArea0 + tmpbc1[:, 1] * tmpArea1
    ay = ay + np.bincount(sub2, val1, minlength=point.shape[0]).reshape(-1, 1)

    val2 = tmpbc0[:, 2] * tmpArea0 + tmpbc1[:, 2] * tmpArea1
    az = az + np.bincount(sub2, val2, minlength=point.shape[0]).reshape(-1, 1)

    ax = ax/b
    ay = ay/b
    az = az/b

    massCenter = np.zeros((N,3),dtype=float)
    massCenter[:, 0] = ax.reshape(-1)
    massCenter[:, 1] = ay.reshape(-1)
    massCenter[:, 2] = az.reshape(-1)
    l1 = np.sum((massCenter - point)*normal,1)
    l = np.matlib.repmat(l1, 3, 1).T

    massCenter = massCenter - l *normal

    return massCenteruu

massCenter = MassCenter(point,cell,normal,gamma=1)
print(massCenter)



f = pl.figure()
axes = a3.Axes3D(f)
mesh.add_plot(axes, showaxis=True)
mesh.find_point(axes, point=massCenter, markersize=100)
pl.show()


