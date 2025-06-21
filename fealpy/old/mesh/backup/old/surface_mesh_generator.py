import numpy as np

from mayavi import mlab

from .TriangleMesh import TriangleMesh

def iso_surface(surface, box, nx=10, ny=10, nz=10):
    X, Y, Z = np.mgrid[
            box[0]:box[1]:(nx+1)*1j,
            box[2]:box[3]:(ny+1)*1j,
            box[4]:box[5]:(nz+1)*1j]

    src = mlab.pipeline.scalar_field(X, Y, Z, surface(X, Y, Z), figure=False)
    obj = mlab.pipeline.iso_surface(src, contours=[0])
    actor = obj.actor.actors[0]
    polyDataObj = actor.mapper.input
    node = np.array(polyDataObj.points)
    node, _ = surface.project(node)
    cell = polyDataObj.polys.data.to_array().reshape(-1, 4)
    cell = cell[:, [1, 3, 2]]

    return TriangleMesh(node, cell)

