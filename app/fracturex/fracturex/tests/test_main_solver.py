
import ipdb
import numpy as np

import pytest
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

from app.fracturex.fracturex.phasefield.main_solve import MainSolve

class TestMainSolver:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_main_solver(self, backend):
        
        mesh = TriangleMesh.from_box([0,1,0,1],1,1)


        E = 200
        nv = 0.3
        params = {'E': E, 'nu': nv, 'Gc': 1.0, 'l0': 0.1}

        ms = MainSolver(mesh=mesh, material_params=params, p=1, method='HybridModel')
        

        ms.add_boundary_condition('force', 'Dirichlet', self.fun1, [0.1, 0.2, 0.3], 'y')
        ms.add_boundary_condition('displacement', 'Dirichlet', self.fun, 0)
        ms.solve(vtkname='test')
        


    
    def fun(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return bm.abs(y) < 1e-10
    
    def fun1(self, p):
        x = p[..., 0]
        y = p[..., 1]
        isindof =  bm.abs(y-1) < 1e-10
        return isindof





if __name__ == "__main__":
    TestMainSolver().test_main_solver('numpy')
    TestMainSolver().test_main_solver('pytorch')



