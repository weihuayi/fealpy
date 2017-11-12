
import os
import subprocess
import tempfile

import meshio
import numpy as np

def generate_mesh(geo_object, dim=3, verbose=True, 
        optimize=True, 
        num_lloyd_steps=100,
        prune_vertices=True):

    # generate the tmp files 
    handle, geo_filename = tempfile.mkstemp(suffix='.geo')
    os.write(handle, geo_object.get_code().encode())
    os.close(handle)
    handle, msh_filename = tempfile.mkstemp(suffix='.msh')
    os.close(handle)

    cmd = [ 'gmsh', '-{}'.format(dim),  geo_filename, '-o', msh_filename]

    if optimize and num_lloyd_steps > 0:
        cmd += ['-optimize_lloyd', str(num_lloyd_steps)]
    # http://stackoverflow.com/a/803421/353337
    p = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT)

    if verbose:
        while True:
            line = p.stdout.readline()
            if not line:
                break
            print(line.decode('utf-8'), end='')

    p.communicate()
    assert p.returncode == 0, \
        'Gmsh exited with error (return code {}).'.format(p.returncode)

    point, cell, point_data, cell_data, field_data = meshio.read(msh_filename, 'gmsh')

    # clean up
    #os.remove(geo_filename)
    #os.remove(msh_filename)

    if prune_vertices:
        # Make sure to include only those vertices which belong to a triangle.
        uvertices, uidx = np.unique(cell['triangle'], return_inverse=True)
        cell = {'triangle': uidx.reshape(cell['triangle'].shape)}
        cell_data = {'triangle': cell_data['triangle']}
        point = point[uvertices]
        for key in point_data:
            point_data[key] = point_data[key][uvertices]

    return point, cell, point_data, cell_data, field_data
