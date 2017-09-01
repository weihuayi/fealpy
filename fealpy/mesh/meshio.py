"""Mesh IO
"""
import scipy.io as sio
from .TriangleMesh import TriangleMesh

def load_mat_mesh(f):
    """ Load mesh in Matlab format
    """
    data = sio.loadmat(f)

    point =data['node']
    cell = data['elem'] - 1

    trimesh = TriangleMesh(point, cell)
    return trimesh

def write_mat_mesh(f, mesh):
    data = {'node':mesh.point, 'elem':mesh.ds.cell+1}
    sio.matlab.savemat(f, data)

def write_mat_linear_system(f, AD, b):
    data = {'AD':AD, 'b':b}
    sio.matlab.savemat(f, data)
