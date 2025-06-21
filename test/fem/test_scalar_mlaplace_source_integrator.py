import ipdb
import numpy as np
import pytest
import sympy as sp

from fealpy.backend import backend_manager as bm
from fealpy.mesh.tetrahedron_mesh import TetrahedronMesh
from fealpy.functionspace.cm_conforming_fe_space3d import CmConformingFESpace3d 
from fealpy.fem import LinearForm 
from fealpy.fem.scalar_mlaplace_source_integrator import ScalarMLaplaceSourceIntegrator
from fealpy.pde.biharmonic_triharmonic_3d import get_flist

from mthlaplace_integrator_data import *


class TestScalarMLaplaceSourceIntegrator:
    @pytest.mark.parametrize("backend", ['numpy','pytorch'])
    #@pytest.mark.parametrize("data", grad_m)
    def test_scalar_mlaplace_source_integrator(self, backend): 
        bm.set_backend(backend)
        def f(p):
            x, y, z = p[..., 0], p[..., 1], p[..., 2]
            return x**4*y**6+z**11
        x = sp.Symbol('x')
        y = sp.Symbol('y')
        z = sp.Symbol('z')
        f = x**4*y**6+z**11
        flist = get_flist(f) 
       
        mesh = TetrahedronMesh.from_box([0,1,0,1,0,1], nx=1, ny=1, nz=1)
        node = mesh.entity('node')
        isCornerNode = bm.zeros(len(node),dtype=bm.bool)
        for n in bm.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[1,1,1],[0,1,1],[0,0,1],[1,0,1]], dtype=bm.float64):
            isCornerNode = isCornerNode | (bm.linalg.norm(node-n[None, :], axis=1)<1e-10)
        space = CmConformingFESpace3d(mesh, p=11, m=1, isCornerNode=isCornerNode)
        gdof = space.number_of_global_dofs()
        lform = LinearForm(space)
        integrator = ScalarMLaplaceSourceIntegrator(2, flist[2], q=14)
        lform.add_integrator(integrator)
        F = lform.assembly()
if __name__ == "__main__":
    #pytest.main(['test_scalar_mlaplace_source_integrator.py', "-q", "-k","test_scalar_mlaplace_source_integrator", "-s"])
    t = TestScalarMLaplaceSourceIntegrator()
    t.test_scalar_mlaplace_source_integrator("numpy")


