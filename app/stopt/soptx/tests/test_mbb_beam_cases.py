import pytest
import numpy as np
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import UniformMesh2d
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from ..cases.mbb_beam_cases import MBBBeamCase
from ..cases.material_properties import MaterialProperties



class TestMBBBeamCase:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_initialize_case_parameters_top88(self, backend):
        bm.set_backend(backend)
        mbb_case = MBBBeamCase(case_name="top88")

        nx = mbb_case.nx
        ny = mbb_case.ny
        
        force = mbb_case.boundary_conditions.force
        dirichlet = mbb_case.boundary_conditions.dirichlet
        is_dirichlet_boundary_edge = mbb_case.boundary_conditions.is_dirichlet_boundary_edge
        is_dirichlet_node = mbb_case.boundary_conditions.is_dirichlet_node
        is_dirichlet_direction = mbb_case.boundary_conditions.is_dirichlet_direction

        assert mbb_case.material_properties is not None, \
            "Material properties should be initialized."
        assert mbb_case.geometry_properties is not None, \
            "Geometry properties should be initialized."
        assert mbb_case.filter_properties is not None, \
            "Filter properties should be initialized."
        assert mbb_case.constraint_conditions is not None, \
            "Constraint conditions should be initialized."
        assert mbb_case.boundary_conditions is not None, \
            "Boundary conditions should be initialized."
        assert mbb_case.termination_criterias is not None, \
            "Termination criterias should be initialized."
        
        extent = [0, nx, 0, ny]
        h = [1, 1]
        origin = [0, 0]
        mesh = UniformMesh2d(extent, h, origin)
        NC = mesh.number_of_cells()

        space_C = LagrangeFESpace(mesh, p=1, ctype='C')
        tensor_space = TensorFunctionSpace(space_C, shape=(-1, 2))
        uh = tensor_space.function()

        F = tensor_space.interpolate(force)
        isDDof = tensor_space.is_boundary_dof(threshold=(is_dirichlet_boundary_edge, 
                                                        is_dirichlet_node,
                                                        is_dirichlet_direction))
        space_D = LagrangeFESpace(mesh, p=0, ctype='D')        
        rho = space_D.function()
        print("-------------")