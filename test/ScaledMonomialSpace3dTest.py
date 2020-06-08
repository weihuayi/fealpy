#!/usr/bin/env python3
# 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from fealpy.functionspace import ScaledMonomialSpace3d
from fealpy.mesh import MeshFactory

class ScaledMonomialSpace3dTest:
    def __init__(self):
        pass
    def one_tet_mesh_test(self, p, plot=True):
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh()
        space = ScaledMonomialSpace3d(mesh, p=p)
        if plot:
            fig = plt.figure()
            axes = fig.gca(projection='3d')
            mesh.add_plot(axes)
            axes.set_axis_off()
            plt.show()

    def diff_index_test(self, p=3):
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh()
        space = ScaledMonomialSpace3d(mesh, p=p)
        index = space.diff_index_1()
        print(index)
        index = space.diff_index_2()
        print(index)

    def face_index_test(self, p=3):
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh()
        space = ScaledMonomialSpace3d(mesh, p=p)
        space.face_diff_index_1()


    def show_frame_test(self):
        fig = plt.figure()
        axes = fig.gca(projection='3d')
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh()
        space = ScaledMonomialSpace3d(mesh, p=1)
        space.show_frame(axes)
        plt.show()

    def show_cell_basis_index_test(self):
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh()
        space = ScaledMonomialSpace3d(mesh, p=3)
        space.show_cell_basis_index(p=3)

    def show_face_basis_index_test(self):
        mfactory = MeshFactory()
        mesh = mfactory.one_tetrahedron_mesh()
        space = ScaledMonomialSpace3d(mesh, p=3)
        space.show_face_basis_index(p=3)

    def face_basis_test(self):
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

    def cell_basis_test(self):
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
#test.one_tet_mesh_test(p=1)
#test.show_frame_test() 
#test.show_face_basis_index_test()
#test.face_basis_test()
#test.cell_basis_test()
#test.diff_index_test(p=3)
#test.show_cell_basis_index_test()
test.face_index_test()
