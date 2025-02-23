
import pytest

from fealpy.backend import backend_manager as bm

from fealpy.pde.poisson_2d import CosCosData 
from fealpy.mesh import TriangleMesh
from fealpy.fem import PoissonLFEMSolver
from fealpy.solver.pangulu import PanguLUSolver

class TestPanguLUSolver:
    @pytest.mark.parametrize('backend', ['numpy'])
    def test(self, backend):
        bm.set_backend(backend)
        pde = CosCosData() 
        domain = pde.domain()
        mesh = TriangleMesh.from_box(box=domain, nx=10, ny=10)
        mesh.uniform_refine(n=2)
        s0 = PoissonLFEMSolver(pde, mesh, 1)

        ds = PanguLUSolver(nb=200, nthread=12)
        x = ds.solve(s0.A, s0.b)
        assert bm.max(bm.abs(s0.A@x - s0.b)) < 1e-12

if __name__ == '__main__':
    pytest.main(["-q"])   
