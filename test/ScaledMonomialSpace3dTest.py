#!/usr/bin/env python3
# 
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from fealpy.functionspace import ScaledMonomialSpace3d
from fealpy.mesh import MeshFactory

class ScaledMonomialSpace3dTest:
    def __init__(self):
        pass
    def one_tet_mesh(self, p, plot=True):
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh()
        space = ScaledMonomialSpace3d(mesh, p=p)
        if plot:
            fig = plt.figure()
            axes = fig.gca(projection='3d')
            mesh.add_plot(axes)
            axes.set_axis_off()
            plt.show()

    def diff_index(self, p=3):
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh()
        space = ScaledMonomialSpace3d(mesh, p=p)
        index = space.diff_index_1()
        print(index)
        index = space.diff_index_2()
        print(index)

    def face_index(self, p=3):
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh()
        space = ScaledMonomialSpace3d(mesh, p=p)
        space.face_diff_index_1()


    def show_frame(self):
        fig = plt.figure()
        axes = fig.gca(projection='3d')
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh()
        space = ScaledMonomialSpace3d(mesh, p=1)
        space.show_frame(axes)
        plt.show()

    def show_cell_basis_index(self, p=1):
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh()
        space = ScaledMonomialSpace3d(mesh, p=1)
        space.show_cell_basis_index(p=p)

    def show_face_basis_index(self):
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh()
        space = ScaledMonomialSpace3d(mesh, p=3)
        space.show_face_basis_index(p=3)

    def face_basis(self):
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh(ttype='iso')
        space = ScaledMonomialSpace3d(mesh, p=1)

        face = mesh.entity('face')
        print(face)
        print('frame:', space.faceframe)
        bc = np.array([[1/3, 1/3, 1/3]])
        bc = np.array([[1, 0, 0]])
        point = mesh.bc_to_point(bc, 'face')
        print(point.shape)
        phi = space.face_basis(point)
        print(phi)

    def cell_basis(self):
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh(ttype='iso')
        space = ScaledMonomialSpace3d(mesh, p=2)

        bc = np.array([[1, 0, 0, 0]])
        point = mesh.bc_to_point(bc, 'cell')
        print(point.shape)
        print(space.cellsize)
        phi = space.basis(point)
        print(phi)






test = ScaledMonomialSpace3dTest()

if sys.argv[1] == "show_cell_basis_index": 
    p = int(sys.argv[2])
    test.show_cell_basis_index(p=p)
