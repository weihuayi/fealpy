#!/usr/bin/env python3
# 

import sys

import numpy as np
import scipy.io as sio


from fealpy.mesh import TriangleMesh, HalfEdgeMesh2d
from fealpy.ccg import ComputationalConformalGeometryAlg
from fealpy.writer import MeshWriter
from fealpy.mesh import CCGMeshReader

class CCGTest():

    def __init__(self):
        self.ccgalg = ComputationalConformalGeometryAlg()


    def tri_cut_graph(self, fname):
        data = sio.loadmat(fname)
        node = np.array(data['node'], dtype=np.float64)
        cell = np.array(data['elem'] - 1, dtype=np.int_)

        mesh = TriangleMesh(node, cell)
        mesh = HalfEdgeMesh2d.from_mesh(mesh, closed=True)
        mesh.ds.NV = 3

        gamma = self.ccgalg.tri_cut_graph(mesh)

        writer = MeshWriter(mesh)
        writer.write(fname='test0.vtu')
        for i, index in enumerate(gamma):
            writer = MeshWriter(mesh, etype='edge', index=index)
            writer.write(fname='test'+str(i+1)+'.vtu')

    def harmonic_map(self, fanme):

        reader = CCGMeshReader(fname)
        tmesh = reader.read()

        mesh = HalfEdgeMesh2d.from_mesh(tmesh, NV=3)
        mesh.nodedata['rgb'] = tmesh.nodedata['rgb']

        cmesh = self.ccgalg.harmonic_map(mesh)
        cmesh.nodedata['rgb'] = tmesh.nodedata['rgb']
        writer = MeshWriter(mesh)
        writer.write(fname='face3.vtu')

        writer = MeshWriter(cmesh)
        writer.write(fname='face2.vtu')


test = CCGTest()

if sys.argv[1] == 'cut_graph':
    fname = sys.argv[2] 
    test.tri_cut_graph(fname)
elif sys.argv[1] == 'harmonic':
    fname = sys.argv[2] 
    test.harmonic_map(fname)
