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
        mesh = mfactory.one_tet_mesh()
        space = ScaledMonomialSpace3d(mesh, p=p)
        if plot:
            fig = plt.figure()
            axes = fig.gca(projection='3d')
            mesh.add_plot(axes)
            axes.set_axis_off()
            plt.show()

    def show_frame_test(self):
        fig = plt.figure()
        axes = fig.gca(projection='3d')

        mfactory = MeshFactory()
        mesh = mfactory.one_tet_mesh()
        space = ScaledMonomialSpace3d(mesh, p=1)
        space.show_frame(axes)
        plt.show()



test = ScaledMonomialSpace3dTest()
#test.one_tet_mesh_test(p=1)
test.show_frame_test()
