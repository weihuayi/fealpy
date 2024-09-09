
import ipdb

import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh

from app.fracturex.fracturex.phasefield.main_solver import MainSolver

class TestMainSolver:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_main_solver(self, backend):
        
        mesh = TriangleMesh.from_box([0,1,0,1],1,1)


        E = 200
        nv = 0.3
        params = {'E': E, 'nu': nv, 'Gc': 1.0, 'l0': 0.1}

        ms = MainSolver(mesh=mesh, material_params=params, p=1, method='HybridModel')
        
        ipoints = mesh.interpolation_points(p=1)

        fixed_ubd = self.fun(ipoints)
        print('fixed_ubd', fixed_ubd)

        force_ubd = self.fun1(ipoints)
        print('force_ubd', force_ubd)

        ms.solve_displacement()
        ms.solve_phase_field()
    
    def fun(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return bm.abs(y) < 1e-10
    
    def fun1(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return bm.abs(y-1) < 1e-10





if __name__ == "__main__":
    TestMainSolver().test_main_solver('numpy')
    TestMainSolver().test_main_solver('pytorch')



