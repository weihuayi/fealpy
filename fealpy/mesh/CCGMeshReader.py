import sys
import numpy as np
import scipy.io as sio

from .TriangleMesh import TriangleMesh

class CCGMeshReader:
    def __init__(self, fname):
        try:
            with open(fname, 'r') as f:
                self.lines = f.read().split('\n')
        except EnvironmentError:
            print("Warning! open file failed!")
        self.cline = 0

    def read(self):
        self.read_vertices()
        self.read_faces()
        self.read_edges()

    def read_vertices(self):
        vertices = list(filter(lambda x : x.find('Vertex') > -1, self.lines))
        print(vertices[0])

    def read_faces(self):
        faces = list(filter(lambda x : x.find('Face') > -1, self.lines))
        print(faces[0])

    def read_edges(self):
        edges = list(filter(lambda x : x.find('Edge') > -1, self.lines))
        print(edges[0])
