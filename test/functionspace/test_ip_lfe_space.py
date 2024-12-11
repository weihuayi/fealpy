from fealpy.functionspace.interior_penalty_fe_space import InteriorPenaltyDof2d, InteriorPenaltyFESpace2d
import pytest
from fealpy.mesh import TriangleMesh
from fealpy.backend import backend_manager as bm
from fealpy.quadrature import Quadrature

class TestInteriorPenaltyDof2d:
    def test_ip_lfe_dof_2d(self):
        mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        dof = InteriorPenaltyDof2d(mesh, 2)
        iedge2celldof = dof.inner_edge_to_cell_dof()
        bedge2celldof = dof.boundary_edge_to_cell_dof()
        print(iedge2celldof)
        print(bedge2celldof)

    def test_boundary_edge_to_cell_dof(self):
        pass

class TestInteriorPenaltyFESpace2d:
    def test_ip_lfe_space_2d(self):
        mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        space = InteriorPenaltyFESpace2d(mesh, p=2)
        qf = mesh.integrator(q=2, etype='edge')
        bcs, ws = qf.get_quadrature_points_and_weights()
        rval = space.grad_normal_jump_basis(bcs)
        print('rv:', rval)
        #space.grad_grad_normal_jump_basis(bcs)

bm.set_backend('numpy')
dof = TestInteriorPenaltyDof2d()
dof.test_ip_lfe_dof_2d()
test_space = TestInteriorPenaltyFESpace2d()
test_space.test_ip_lfe_space_2d()