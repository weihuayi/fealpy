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

def write_mat_mesh(mesh, f):
    data = {'node':mesh.point, 'elem':mesh.ds.cell+1}
    sio.matlab.savemat(f, data)
