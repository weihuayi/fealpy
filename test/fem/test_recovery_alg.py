
import ipdb
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import RecoveryAlg

class TestRecoveryAlg:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_recovery_alg(self, backend): 
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        space = LagrangeFESpace(mesh, 2)
        uh = space.function()

        uh[:] = self.fun(space.interpolation_points())
        recovery = RecoveryAlg()
        eta0 = recovery.recovery_estimate(uh, method='simple')
        eta1 = recovery.recovery_estimate(uh, method='area_harmonic')
        eta2 = recovery.recovery_estimate(uh, method='area')
        eta3 = recovery.recovery_estimate(uh, method='distance')
        eta4 = recovery.recovery_estimate(uh, method='distance_harmonic')

        print('eta0:', eta0)
        print('eta1:', eta1)
        print('eta2:', eta2)
        print('eta3:', eta3)
        print('eta4:', eta4)
        

    def fun(self, p):
        x = p[..., 0]
        y = p[..., 1]
        f = x*y
        bm.set_at(f, 1, 10)
        #f[1] = 10
        return f


if __name__ == "__main__":
    TestRecoveryAlg().test_recovery_alg('numpy')
    TestRecoveryAlg().test_recovery_alg('pytorch')
    TestRecoveryAlg().test_recovery_alg('jax')
#    pytest.main(['test_recovery_alg.py', "-q"])   

