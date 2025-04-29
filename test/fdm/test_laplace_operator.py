from fealpy.fdm import Laplace
from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh1d, UniformMesh2d, UniformMesh3d
import pytest

class TestLaplaceOperator:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_1d_operator(self, backend):
        bm.set_backend(backend)

        class SinPDEData:
            def domain(self):
                return [0, 1]

            def solution(self, p):
                return bm.sin(4*bm.pi*p)

            def source(self, p):
                return 16*bm.pi**2*bm.sin(4*bm.pi*p)
  
            def dirichlet(self, p):
                return self.solution(p)

        pde = SinPDEData()
        domain = pde.domain()

        nx = 20
        hx = (domain[1] - domain[0])/nx
        mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

        maxit = 5
        em = bm.zeros((3, maxit), dtype=bm.float64)
        for i in range(maxit):
            a = Laplace(mesh, pde)
            uh = a.solve()
            em[0, i], em[1, i], em[2, i] = a.error(pde.solution, uh)

            if i < maxit:
                mesh.uniform_refine()

        print("em_ratio:", em[:, 0:-1]/em[:, 1:])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_2d_operator(self, backend):
        bm.set_backend(backend)

        class SinSinPDEData:
            def domain(self):
                return bm.array([0, 1, 0, 1])

            def solution(self, p):
                x = p[..., 0]
                y = p[..., 1]
                pi = bm.pi
                val = bm.sin(pi*x)*bm.sin(pi*y)
                return val 

            def source(self, p):
                x = p[..., 0]
                y = p[..., 1]
                pi = bm.pi
                val = 2*pi*pi*bm.sin(pi*x)*bm.sin(pi*y)
                return val

            def dirichlet(self, p):
                return self.solution(p)

        pde = SinSinPDEData()
        domain = pde.domain()

        nx = 10
        ny = 10
        hx = (domain[1] - domain[0])/nx
        hy = (domain[3] - domain[2])/ny

        mesh = UniformMesh2d((0, nx, 0, ny), h=(hx, hy), origin=(domain[0], domain[2]))

        maxit = 5
        em = bm.zeros((3, maxit), dtype=bm.float64)
        for i in range(maxit):
            a = Laplace(mesh, pde)
            uh = a.solve()
            em[0, i], em[1, i], em[2, i] = a.error(pde.solution, uh)

            if i < maxit:
                mesh.uniform_refine()
        print("em_ratio:", em[:, 0:-1]/em[:, 1:])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_3d_operator(self, backend):
        bm.set_backend(backend)

        class SinSinSinPDEData:
            def domain(self):
                return bm.array([0, 1, 0, 1, 0, 1])

            def solution(self, p):
                x = p[..., 0]
                y = p[..., 1]
                z = p[..., 2]
                pi = bm.pi
                val = bm.sin(pi*x)*bm.sin(pi*y)*bm.sin(pi*z)
                return val 

            def source(self, p):
                x = p[..., 0]
                y = p[..., 1]
                z = p[..., 2]
                pi = bm.pi
                val = 3 * pi**2 * bm.sin(pi*x) * bm.sin(pi*y) * bm.sin(pi*z)
                return val

            def dirichlet(self, p):
                return self.solution(p)

        pde = SinSinSinPDEData()
        domain = pde.domain()

        nx = 5
        ny = 5
        nz = 5

        hx = (domain[1] - domain[0])/nx
        hy = (domain[3] - domain[2])/ny
        hz = (domain[5] - domain[4])/nz

        mesh = UniformMesh3d((0, nx, 0, ny, 0, nz), h=(hx, hy, hz), origin=(domain[0], domain[2], domain[4]))

        maxit = 1
        em = bm.zeros((3, maxit), dtype=bm.float64)
        for i in range(maxit):
            a = Laplace(mesh, pde)
            uh = a.solve()
            em[0, i], em[1, i], em[2, i] = a.error(pde.solution, uh)

            if i < maxit:
                mesh.uniform_refine()

        print("em_ratio:", em[:, 0:-1]/em[:, 1:])
