
import pytest
import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.solver import GAMGSolver 
from fealpy.sparse import csr_matrix,coo_matrix
from fealpy.functionspace import LagrangeFESpace
from gamg_solver_data import * 



class TestGAMGSolverInterfaces:
    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("data", init_data)
    def test_init(self, data, backend):
        bm.set_backend(backend)
        solver = GAMGSolver(**data) 
        assert solver is not None
        assert solver.maxit == data['maxit']

    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("data", init_data)
    @pytest.mark.parametrize("test_data",test_data)
    def test_vcycle(self,data,test_data, backend):
        bm.set_backend(backend)
        solver = GAMGSolver(**data) 
        A = test_data['A']
        p = test_data['p']
        domain = test_data['domain']
        nx = test_data['nx']
        ny = test_data['ny']
        
        mesh = TriangleMesh.from_box(box=domain,nx=nx,ny=ny)
        space = LagrangeFESpace(mesh, p=p)

        m = test_data['m']
        P = mesh.uniform_refine(n=m,returnim=True)
        cdegree = list(range(1,p))
        solver.setup(A=A,P=P,space=space,cdegree=cdegree)
        f = test_data['f']
        x,info = solver.solve(f)
        atol = 1e-8
        rtol = 1e-8

        # 断言确保 solver 不为空
        assert solver is not None
        f_norm = np.linalg.norm(f)
        #是否收敛
        assert info['residual'] < atol * f_norm or info['residual'] < rtol

    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("data", init_data)
    @pytest.mark.parametrize("test_data",test_data)
    def test_wcycle(self,data,test_data, backend):
        bm.set_backend(backend)
        data['ptype'] = 'W'
        solver = GAMGSolver() 
        A = test_data['A']
        p = test_data['p']
        domain = test_data['domain']
        nx = test_data['nx']
        ny = test_data['ny']
        
        mesh = TriangleMesh.from_box(box=domain,nx=nx,ny=ny)
        space = LagrangeFESpace(mesh, p=p)
        
        m = test_data['m']
        P = mesh.uniform_refine(n=m,returnim=True)
        cdegree = list(range(1,p))
        solver.setup(A=A,P=P,space=space,cdegree=cdegree)
        f = test_data['f']
        x,info = solver.solve(f)
        atol = 1e-8
        rtol = 1e-8

        # 断言确保 solver 不为空
        assert solver is not None
        f_norm = np.linalg.norm(f)
        #是否收敛
        assert info['residual'] < atol * f_norm or info['residual'] < rtol

    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("data", init_data)
    @pytest.mark.parametrize("test_data",test_data)
    def test_fcycle(self,data,test_data, backend):
        bm.set_backend(backend)
        solver = GAMGSolver()
        data['ptype'] = 'F'
        A = test_data['A']
        p = test_data['p']
        domain = test_data['domain']
        nx = test_data['nx']
        ny = test_data['ny']
        
        mesh = TriangleMesh.from_box(box=domain,nx=nx,ny=ny)
        space = LagrangeFESpace(mesh, p=p)
        
        m = test_data['m']
        P = mesh.uniform_refine(n=m,returnim=True)
        cdegree = list(range(1,p))
        solver.setup(A=A,P=P,space=space,cdegree=cdegree)
        f = test_data['f']
        x,info = solver.solve(f)
        atol = 1e-8
        rtol = 1e-8

        # 断言确保 solver 不为空
        assert solver is not None
        f_norm = np.linalg.norm(f)
        #是否收敛
        assert info['residual'] < atol * f_norm or info['residual'] < rtol    




if __name__ == "__main__":
    pytest.main(["./test_gamg_solver.py",'-k' ,"test_vcycle"])
    test = TestGAMGSolverInterfaces()