
import os
import subprocess
import tempfile

import meshio
import numpy as np

def generate_mesh(geo_object, dim=3, prune_vertices=True):

    # generate the tmp files 
    handle, geo_filename = tempfile.mkstemp(suffix='.geo')
    os.write(handle, geo_object.get_code().encode())
    os.close(handle)
    handle, msh_filename = tempfile.mkstemp(suffix='.msh')
    os.close(handle)

    cmd = [ 'gmsh', '-{}'.format(dim),  geo_filename, '-o', msh_filename]

    # http://stackoverflow.com/a/803421/353337
    p = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT)

    p.communicate()
    assert p.returncode == 0, \
        'Gmsh exited with error (return code {}).'.format(p.returncode)

    point, cell, point_data, cell_data, field_data = meshio.read(msh_filename)

    # clean up
    os.remove(geo_filename)
    os.remove(msh_filename)

    if prune_vertices:
        # Make sure to include only those vertices which belong to a triangle.
        uvertices, uidx = numpy.unique(cells['triangle'], return_inverse=True)
        cells = {'triangle': uidx.reshape(cells['triangle'].shape)}
        cell_data = {'triangle': cell_data['triangle']}
        X = X[uvertices]
        for key in pt_data:
            pt_data[key] = pt_data[key][uvertices]

    return X, cells, pt_data, cell_data, field_data
