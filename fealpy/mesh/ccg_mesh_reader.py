import sys
import numpy as np
import scipy.io as sio
import re

from .triangle_mesh import TriangleMesh

class CCGMeshReader:
    def __init__(self, fname):
        try:
            with open(fname, 'r') as f:
                self.lines = f.read().split('\n')
        except EnvironmentError:
            print("Warning! open file failed!")
        self.cline = 0

    def read(self):
        data, uv, rgb = self.read_vertices()
        cell = self.read_faces()
        self.read_edges()

        NN = len(data)
        idxmap = np.zeros(int(data[:, 0].max()), dtype=np.int_)
        idxmap[data[:, 0].astype(np.int_)-1] = range(NN)
        cell = idxmap[cell-1]

        mesh = TriangleMesh(data[:, 1:].copy(), cell)

        if uv is not None:
            mesh.nodedata['uv'] = uv

        if rgb is not None:
            mesh.nodedata['rgb'] = rgb

        return mesh

    def read_vertices(self):
        vertices = list(filter(lambda x : x.find('Vertex') > -1, self.lines))
        data = np.array([ s[7:s.find(' {')].split(' ') for s in vertices],
                dtype=np.float64)
        if vertices[0].find('uv')>-1:
            uv = [re.findall(r'uv=\(.*?\)', s)[0][4:-1].split(' ') for s in vertices]
            uv = np.array(uv, dtype=np.float64) 
        else:
            uv = None

        if vertices[0].find('rgb') > -1:
            rgb = [ re.findall(r'rgb=\(.*?\)', s)[0][5:-1].split(' ') for s in vertices]
            rgb = np.array(rgb, dtype=np.float64) 
        else:
            rgb = None

        return data, uv, rgb

    def read_faces(self):
        faces = list(filter(lambda x : x.find('Face') > -1, self.lines))
        cell = np.array([s[5:].split(' ')[1:] for s in faces], dtype=np.int_)
        return cell

    def read_edges(self):
        edges = list(filter(lambda x : x.find('Edge') > -1, self.lines))
        print(edges[0])
